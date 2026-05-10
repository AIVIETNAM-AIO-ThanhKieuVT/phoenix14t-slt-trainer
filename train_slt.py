"""
Trainer for an independent Sign Language Transformer (SLT) on PHOENIX-2014T.

Designed to avoid the torchtext.legacy / slt-Camgoz dependency hell on modern
Python. Runs on standard PyTorch >= 2.0.

Usage (Colab or any Python 3.10-3.12 + GPU):
    python train_slt.py \\
        --train slt_data/phoenix14t.train.pickle.gzip \\
        --dev   slt_data/phoenix14t.dev.pickle.gzip \\
        --test  slt_data/phoenix14t.test.pickle.gzip \\
        --config config_seed142.yaml \\
        --model_dir models/slt_seed142_4layer_h384

The script auto-resumes from `latest.ckpt` if it exists in `model_dir`.
"""
from __future__ import annotations

import argparse
import gzip
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from back_translation.bt_model import build_model  # noqa: E402

PAD, UNK, BOS, EOS, SIL = "<pad>", "<unk>", "<s>", "</s>", "<si>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocab:
    """joeynmt-compatible Vocabulary (writes one token per line)."""

    def __init__(self, tokens: List[str]):
        # Deduplicate while preserving order
        seen = set()
        self.itos = []
        for t in tokens:
            if t not in seen:
                self.itos.append(t)
                seen.add(t)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def from_freqs(cls, specials: List[str], freqs: Counter, min_freq: int = 1):
        words = [w for w, f in freqs.most_common() if f >= min_freq]
        return cls(specials + words)

    def encode(self, words: List[str]) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(w, unk) for w in words]

    def write(self, path: Path):
        with open(path, "w") as f:
            for tok in self.itos:
                f.write(tok + "\n")

    def __len__(self):
        return len(self.itos)

    @property
    def pad(self):
        return self.stoi[PAD]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_pickle(path: Path):
    if str(path).endswith(".gzip") or str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def build_vocabs(train_items, min_freq: int = 1):
    gls_freqs, txt_freqs = Counter(), Counter()
    for it in train_items:
        gls_freqs.update(it["gloss"].split())
        txt_freqs.update(it["text"].split())
    gls_vocab = Vocab.from_freqs([SIL, UNK, PAD], gls_freqs, min_freq=min_freq)
    txt_vocab = Vocab.from_freqs([UNK, PAD, BOS, EOS], txt_freqs, min_freq=min_freq)
    return gls_vocab, txt_vocab


class SignDataset(Dataset):
    def __init__(self, items, gls_vocab, txt_vocab, subsample: int = 2,
                 max_sent_length: int = 400):
        self.items = items
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        self.subsample = subsample
        self.max_sent_length = max_sent_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        sign = it["sign"]
        if not isinstance(sign, torch.Tensor):
            sign = torch.tensor(sign, dtype=torch.float32)
        else:
            sign = sign.float()
        if self.subsample > 1:
            sign = sign[:: self.subsample]
        if sign.shape[0] > self.max_sent_length:
            sign = sign[: self.max_sent_length]

        gls_ids = torch.tensor(
            self.gls_vocab.encode(it["gloss"].split()), dtype=torch.long
        )
        txt_words = it["text"].split()
        txt_ids = torch.tensor(
            [self.txt_vocab.stoi[BOS]]
            + self.txt_vocab.encode(txt_words)
            + [self.txt_vocab.stoi[EOS]],
            dtype=torch.long,
        )
        return {
            "sign": sign,
            "gls": gls_ids,
            "txt": txt_ids,
            "name": it["name"],
            "text": it["text"],
        }


def collate(samples, txt_pad: int, gls_pad: int):
    samples = sorted(samples, key=lambda s: s["sign"].shape[0], reverse=True)
    sgn_lengths = torch.tensor([s["sign"].shape[0] for s in samples], dtype=torch.long)
    gls_lengths = torch.tensor([s["gls"].shape[0] for s in samples], dtype=torch.long)
    txt_lengths = torch.tensor([s["txt"].shape[0] for s in samples], dtype=torch.long)

    B = len(samples)
    max_sgn = int(sgn_lengths.max())
    max_gls = int(gls_lengths.max())
    max_txt = int(txt_lengths.max())
    feat_dim = samples[0]["sign"].shape[1]

    sgn_pad = torch.zeros(B, max_sgn, feat_dim)
    for i, s in enumerate(samples):
        n = s["sign"].shape[0]
        sgn_pad[i, :n] = s["sign"]

    gls_pad_t = torch.full((B, max_gls), gls_pad, dtype=torch.long)
    for i, s in enumerate(samples):
        n = s["gls"].shape[0]
        if n > 0:
            gls_pad_t[i, :n] = s["gls"]

    txt_pad_t = torch.full((B, max_txt), txt_pad, dtype=torch.long)
    for i, s in enumerate(samples):
        n = s["txt"].shape[0]
        txt_pad_t[i, :n] = s["txt"]

    sgn_mask = (torch.arange(max_sgn)[None, :] < sgn_lengths[:, None]).unsqueeze(1)

    # joeynmt convention: txt_input = decoder input ([BOS, w1, ..., wN-1])
    # txt = target for loss ([w1, ..., wN-1, EOS])
    txt_input = txt_pad_t[:, :-1].contiguous()
    txt_target = txt_pad_t[:, 1:].contiguous()
    txt_input_mask = (txt_input != txt_pad).unsqueeze(1)

    return SimpleNamespace(
        sgn=sgn_pad,
        sgn_mask=sgn_mask,
        sgn_lengths=sgn_lengths,
        gls=gls_pad_t,
        gls_lengths=gls_lengths,
        txt=txt_target,
        txt_input=txt_input,
        txt_mask=txt_input_mask,
        txt_lengths=txt_lengths,
        names=[s["name"] for s in samples],
        ref_texts=[s["text"] for s in samples],
    )


def to_device(batch, device):
    for k, v in batch.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(batch, k, v.to(device))
    return batch


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class XentLoss(nn.Module):
    """Cross-entropy on log_probs, ignoring pad."""

    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, log_probs, target):
        B, T, V = log_probs.shape
        return F.nll_loss(
            log_probs.reshape(-1, V),
            target.reshape(-1),
            ignore_index=self.pad_idx,
            reduction="sum",
        )


class LabelSmoothingLoss(nn.Module):
    """Label smoothing on log_probs, ignoring pad."""

    def __init__(self, pad_idx: int, smoothing: float):
        super().__init__()
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, log_probs, target):
        if self.smoothing <= 0.0:
            B, T, V = log_probs.shape
            return F.nll_loss(
                log_probs.reshape(-1, V),
                target.reshape(-1),
                ignore_index=self.pad_idx,
                reduction="sum",
            )

        B, T, V = log_probs.shape
        log_probs = log_probs.reshape(-1, V)
        target = target.reshape(-1)

        # NLL for true labels
        nll = F.nll_loss(
            log_probs,
            target,
            ignore_index=self.pad_idx,
            reduction="sum",
        )

        # Uniform smoothing over non-pad tokens
        smooth = -log_probs.mean(dim=-1)
        non_pad = target.ne(self.pad_idx)
        smooth = smooth[non_pad].sum()

        eps = self.smoothing
        return (1.0 - eps) * nll + eps * smooth


class CtcLoss(nn.Module):
    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="sum")

    def forward(self, log_probs, target, input_lengths, target_lengths):
        # log_probs comes in as [T, B, V] from bt_model.forward
        return self.ctc(log_probs, target, input_lengths, target_lengths)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(model, batch, txt_vocab, max_len: int = 30):
    model.eval()
    bos = txt_vocab.stoi[BOS]
    eos = txt_vocab.stoi[EOS]
    pad = txt_vocab.stoi[PAD]
    device = batch.sgn.device

    encoder_output, encoder_hidden = model.encode(
        sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
    )
    B = batch.sgn.shape[0]
    ys = torch.full((B, 1), bos, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        trg_mask = torch.ones(B, 1, ys.shape[1], dtype=torch.bool, device=device)
        decoder_outputs = model.decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            sgn_mask=batch.sgn_mask,
            txt_input=ys,
            unroll_steps=ys.shape[1],
            txt_mask=trg_mask,
        )
        word_outputs = decoder_outputs[0]  # [B, T, V]
        next_tok = word_outputs[:, -1, :].argmax(dim=-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos)
        if finished.all():
            break
    return ys[:, 1:]  # strip BOS


def ids_to_words(ids, vocab):
    eos = vocab.stoi[EOS]
    pad = vocab.stoi[PAD]
    out = []
    for tok_id in ids.tolist():
        if tok_id == eos:
            break
        if tok_id == pad:
            continue
        out.append(vocab.itos[tok_id])
    return out


def compute_bleu4(hyps: List[str], refs: List[str]) -> float:
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(hyps, [refs]).score
    except Exception:
        # Fallback: very rough overlap score
        return float("nan")


def validate(model, dev_loader, txt_vocab, device, max_len: int,
             beam_size: int = 1, beam_alpha: float = -1):
    model.eval()
    hyps, refs = [], []
    for batch in dev_loader:
        batch = to_device(batch, device)
        if beam_size <= 1:
            out_ids = greedy_decode(model, batch, txt_vocab, max_len=max_len)
            for i in range(out_ids.shape[0]):
                hyp_words = ids_to_words(out_ids[i], txt_vocab)
                hyps.append(" ".join(hyp_words))
                refs.append(batch.ref_texts[i])
        else:
            _, stacked_txt_output, _ = model.run_batch(
                batch,
                translation_beam_size=beam_size,
                translation_beam_alpha=beam_alpha,
                translation_max_output_length=max_len,
            )
            for i in range(stacked_txt_output.shape[0]):
                hyp_words = ids_to_words(
                    torch.tensor(stacked_txt_output[i], device=device),
                    txt_vocab,
                )
                hyps.append(" ".join(hyp_words))
                refs.append(batch.ref_texts[i])
    return compute_bleu4(hyps, refs), hyps[:3], refs[:3]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_ckpt(path: Path, model, optimizer, scheduler, epoch: int, best_bleu: float):
    sd = model.state_dict() if not hasattr(model, "module") else model.module.state_dict()
    state = {
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_bleu": best_bleu,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def load_ckpt(path: Path, model, optimizer, scheduler, device):
    state = torch.load(path, map_location=device, weights_only=False)
    sd = state["model_state"]
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(sd, strict=True)
    if optimizer is not None and state.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(state["optimizer_state"])
        except Exception as e:
            print(f"[warn] could not restore optimizer: {e}")
    if scheduler is not None and state.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(state["scheduler_state"])
        except Exception as e:
            print(f"[warn] could not restore scheduler: {e}")
    return state.get("epoch", 0), state.get("best_bleu", 0.0)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", default=None, help="Optional, kept for compatibility.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["training"].get("random_seed", 142)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] model_dir: {model_dir}")

    device = torch.device(args.device)
    print(f"[setup] device: {device}")

    # Load data
    print(f"[data] loading {args.train}")
    train_items = load_pickle(args.train)
    print(f"[data] loading {args.dev}")
    dev_items = load_pickle(args.dev)
    print(f"[data] train={len(train_items)}, dev={len(dev_items)}")

    # Build / load vocabs
    gls_vocab_path = model_dir / "gls.vocab"
    txt_vocab_path = model_dir / "txt.vocab"
    if gls_vocab_path.exists() and txt_vocab_path.exists():
        with open(gls_vocab_path) as f:
            gls_vocab = Vocab([line.rstrip() for line in f])
        with open(txt_vocab_path) as f:
            txt_vocab = Vocab([line.rstrip() for line in f])
        print(f"[vocab] loaded existing: |gls|={len(gls_vocab)}, |txt|={len(txt_vocab)}")
    else:
        gls_vocab, txt_vocab = build_vocabs(train_items, min_freq=1)
        gls_vocab.write(gls_vocab_path)
        txt_vocab.write(txt_vocab_path)
        print(f"[vocab] built+saved: |gls|={len(gls_vocab)}, |txt|={len(txt_vocab)}")

    # Save config (so eval pipeline can read it)
    cfg["training"]["model_dir"] = str(model_dir)
    with open(model_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    subsample = cfg["data"].get("skeleton_subsample", 2)
    train_ds = SignDataset(train_items, gls_vocab, txt_vocab, subsample=subsample)
    dev_ds = SignDataset(dev_items, gls_vocab, txt_vocab, subsample=subsample)

    batch_size = cfg["training"].get("batch_size", 32)
    txt_pad, gls_pad = txt_vocab.pad, gls_vocab.pad

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate(b, txt_pad, gls_pad),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate(b, txt_pad, gls_pad),
    )

    # Build model
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    feat_size = cfg["data"]["feature_size"]
    if isinstance(feat_size, list):
        feat_size = sum(feat_size)

    # bt_model.build_model needs vocab.stoi, vocab.itos, and len(vocab) — our
    # Vocab class already provides all three, so pass it directly.
    model = build_model(
        cfg=cfg["model"],
        sgn_dim=feat_size,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] built, {n_params/1e6:.1f}M params, "
          f"do_recognition={do_recognition}, do_translation={do_translation}")

    # Loss + optimizer
    smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    xent = LabelSmoothingLoss(txt_pad, smoothing).to(device)
    ctc = CtcLoss(blank=gls_vocab.stoi[SIL]).to(device)
    rec_w = cfg["training"].get("recognition_loss_weight", 1.0)
    trans_w = cfg["training"].get("translation_loss_weight", 1.0)

    lr = float(cfg["training"].get("learning_rate", 1e-3))
    wd = float(cfg["training"].get("weight_decay", 1e-3))
    betas = cfg["training"].get("betas", [0.9, 0.998])
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=tuple(betas), weight_decay=wd)

    scheduling = cfg["training"].get("scheduling", "plateau").lower()
    if scheduling == "noam":
        warmup = int(cfg["training"].get("warmup_steps", 4000))
        d_model = cfg["model"]["encoder"]["hidden_size"]
        factor = float(cfg["training"].get("learning_rate", 1.0))
        optimizer.param_groups[0]["lr"] = 1.0

        def _noam(step):
            step = max(step, 1)
            return factor * (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, _noam)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg["training"].get("decrease_factor", 0.8),
            patience=cfg["training"].get("patience", 5),
            min_lr=float(cfg["training"].get("learning_rate_min", 1e-8)),
        )

    # Resume
    latest_path = model_dir / "latest.ckpt"
    best_path = model_dir / "best.ckpt"
    val_log_path = model_dir / "validations.txt"

    start_epoch = 0
    best_bleu = 0.0
    if latest_path.exists():
        start_epoch, best_bleu = load_ckpt(latest_path, model, optimizer, scheduler, device)
        print(f"[resume] from epoch {start_epoch}, best BLEU so far = {best_bleu:.2f}")
    else:
        print("[resume] no latest.ckpt found, starting fresh")

    # Training
    n_epochs = cfg["training"].get("epochs", 100)
    val_freq = cfg["training"].get(
        "validation_freq_epochs",
        cfg["training"].get("validation_freq", 1),
    )
    log_every = cfg["training"].get("logging_freq", 100)
    max_decode_len = cfg["training"].get("translation_max_output_length", 30)
    eval_beam = int(cfg["training"].get("eval_translation_beam_size", 1))
    eval_alpha = float(cfg["training"].get("eval_translation_beam_alpha", -1))
    grad_accum = int(cfg["training"].get("batch_multiplier", 1))
    if grad_accum < 1:
        grad_accum = 1
    global_step = 0

    for epoch in range(start_epoch, n_epochs):
        model.train()
        t_epoch = time.time()
        running_loss = 0.0
        running_n = 0

        for step, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            if step % grad_accum == 0:
                optimizer.zero_grad()

            decoder_outputs, gloss_log_probs = model(
                sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_lengths=batch.sgn_lengths,
                txt_input=batch.txt_input, txt_mask=batch.txt_mask,
            )

            loss = batch.sgn.new_zeros(())

            if do_translation and decoder_outputs is not None:
                word_outputs = decoder_outputs[0]
                txt_log_probs = F.log_softmax(word_outputs, dim=-1)
                trans_loss = xent(txt_log_probs, batch.txt) * trans_w
                # Normalize per token (excluding pad) for stable log values
                n_tok = (batch.txt != txt_pad).sum().clamp(min=1)
                loss = loss + trans_loss / n_tok

            if do_recognition and gloss_log_probs is not None:
                ctc_loss = ctc(
                    gloss_log_probs,                  # [T, B, V] log-softmax
                    batch.gls,                         # [B, T_gls]
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                ) * rec_w
                loss = loss + ctc_loss / batch.sgn.shape[0]

            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                global_step += 1
                if scheduling == "noam":
                    scheduler.step()

            running_loss += loss.item()
            running_n += 1
            if step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"  Ep{epoch} [{step}/{len(train_loader)}] "
                      f"loss={loss.item():.3f}  avg={running_loss/running_n:.3f}  lr={lr_now:.2e}")

        # End of epoch — validate
        if (epoch + 1) % val_freq == 0:
            t_val = time.time()
            bleu, sample_hyps, sample_refs = validate(
                model, dev_loader, txt_vocab, device, max_decode_len,
                beam_size=eval_beam, beam_alpha=eval_alpha,
            )
            if scheduling != "noam":
                scheduler.step(bleu)
            line = (f"epoch={epoch}  bleu4={bleu:.2f}  best={max(bleu, best_bleu):.2f}  "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                    f"epoch_time={time.time()-t_epoch:.1f}s  val_time={time.time()-t_val:.1f}s")
            print(line)
            with open(val_log_path, "a") as f:
                f.write(line + "\n")
                for h, r in zip(sample_hyps, sample_refs):
                    f.write(f"  REF: {r}\n  HYP: {h}\n")

            if bleu > best_bleu:
                best_bleu = bleu
                save_ckpt(best_path, model, optimizer, scheduler, epoch, best_bleu)
                print(f"  ✅ new best BLEU={best_bleu:.2f}, saved {best_path.name}")

        # Always save latest for resume
        save_ckpt(latest_path, model, optimizer, scheduler, epoch + 1, best_bleu)

    print(f"\n[done] training finished. Best dev BLEU-4 = {best_bleu:.2f}")
    print(f"[done] model_dir = {model_dir}")


if __name__ == "__main__":
    main()
