"""
Evaluate a trained SLT checkpoint with beam search.

Usage:
    python eval_slt.py \\
        --config models/slt_seed142_3layer_h256/config.yaml \\
        --model_dir models/slt_seed142_3layer_h256 \\
        --data slt_data/phoenix14t.dev.pickle.gzip \\
        --ckpt best.ckpt \\
        --beam_size 3 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_slt import (  # noqa: E402
    BOS, EOS, PAD,
    Vocab, SignDataset, collate, load_pickle, to_device,
)
from back_translation.bt_model import build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------
@torch.no_grad()
def beam_search_decode(model, batch, txt_vocab, beam_size: int = 3,
                       max_len: int = 30, length_penalty: float = 1.0):
    """GNMT-style beam search. Returns (B, T_out) token-id tensor (no BOS)."""
    model.eval()
    bos = txt_vocab.stoi[BOS]
    eos = txt_vocab.stoi[EOS]
    pad = txt_vocab.stoi[PAD]
    device = batch.sgn.device
    B = batch.sgn.shape[0]
    K = beam_size

    encoder_output, encoder_hidden = model.encode(
        sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
    )

    T_enc, D = encoder_output.shape[1], encoder_output.shape[2]
    encoder_output = (
        encoder_output.unsqueeze(1).expand(B, K, T_enc, D).reshape(B * K, T_enc, D)
    )
    sgn_mask_e = (
        batch.sgn_mask.unsqueeze(1)
        .expand(B, K, *batch.sgn_mask.shape[1:])
        .reshape(B * K, *batch.sgn_mask.shape[1:])
    )

    ys = torch.full((B * K, 1), bos, dtype=torch.long, device=device)
    beam_scores = torch.zeros(B, K, device=device)
    beam_scores[:, 1:] = float("-inf")
    beam_scores = beam_scores.view(-1)
    finished = torch.zeros(B * K, dtype=torch.bool, device=device)

    for _ in range(max_len):
        trg_mask = torch.ones(B * K, 1, ys.shape[1], dtype=torch.bool, device=device)
        decoder_outputs = model.decode(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            sgn_mask=sgn_mask_e,
            txt_input=ys,
            unroll_steps=ys.shape[1],
            txt_mask=trg_mask,
        )
        word_outputs = decoder_outputs[0]                       # [B*K, T, V]
        log_probs = F.log_softmax(word_outputs[:, -1, :], -1)   # [B*K, V]
        V = log_probs.shape[-1]

        # Finished beams: only PAD with score 0 to keep length stable
        log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
        log_probs[finished, pad] = 0.0

        next_scores = (beam_scores.unsqueeze(-1) + log_probs).view(B, K * V)
        topk_scores, topk_idx = next_scores.topk(K, dim=-1)
        beam_idx = topk_idx // V
        token_idx = topk_idx % V

        global_beam_idx = (
            beam_idx + torch.arange(B, device=device).unsqueeze(-1) * K
        ).view(-1)
        ys = ys[global_beam_idx]
        ys = torch.cat([ys, token_idx.view(-1, 1)], dim=1)
        finished = finished[global_beam_idx] | (token_idx.view(-1) == eos)
        beam_scores = topk_scores.view(-1)

        if finished.all():
            break

    # Length-penalty re-scoring
    not_pad = (ys != pad).sum(-1).float() - 1
    not_pad = not_pad.clamp(min=1)
    lp = ((5 + not_pad) / 6) ** length_penalty
    final_scores = (beam_scores / lp).view(B, K)
    best_beam = final_scores.argmax(dim=-1)
    best_global = best_beam + torch.arange(B, device=device) * K
    best_ys = ys[best_global]
    return best_ys[:, 1:]                                       # strip BOS


# ---------------------------------------------------------------------------
# Token -> string
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def levenshtein(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return m
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = tmp
    return dp[m]


def compute_wer(refs, hyps):
    total_edits = 0
    total_words = 0
    for r, h in zip(refs, hyps):
        r_w = r.split()
        h_w = h.split()
        total_edits += levenshtein(r_w, h_w)
        total_words += len(r_w)
    return 100.0 * total_edits / max(1, total_words)


def compute_bleu(hyps, refs):
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs], force=True).score


def compute_chrf(hyps, refs):
    import sacrebleu
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def compute_rouge_l(hyps, refs):
    """Crude corpus ROUGE-L F1 (whitespace tokens)."""
    def _lcs(a, b):
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]
    f1s = []
    for h, r in zip(hyps, refs):
        hw, rw = h.split(), r.split()
        if not hw or not rw:
            f1s.append(0.0); continue
        lcs = _lcs(hw, rw)
        p = lcs / len(hw)
        rc = lcs / len(rw)
        f1s.append(2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0)
    return 100.0 * sum(f1s) / len(f1s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Either training config or model_dir/config.yaml")
    ap.add_argument("--model_dir", required=True,
                    help="Where best.ckpt + vocabs live")
    ap.add_argument("--data", required=True,
                    help="pickle.gzip with sign/text/gloss items")
    ap.add_argument("--ckpt", default="best.ckpt")
    ap.add_argument("--beam_size", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None,
                    help="Optional path to dump (name, ref, hyp) lines for inspection")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    print(f"[setup] device: {device}")
    print(f"[setup] beam_size: {args.beam_size}  length_penalty: {args.length_penalty}")

    # Vocabs
    with open(model_dir / "gls.vocab") as f:
        gls_vocab = Vocab([line.rstrip() for line in f])
    with open(model_dir / "txt.vocab") as f:
        txt_vocab = Vocab([line.rstrip() for line in f])
    print(f"[vocab] |gls|={len(gls_vocab)}, |txt|={len(txt_vocab)}")

    # Data
    print(f"[data] loading {args.data}")
    items = load_pickle(args.data)
    print(f"[data] items={len(items)}")

    subsample = cfg["data"].get("skeleton_subsample", 2)
    ds = SignDataset(items, gls_vocab, txt_vocab, subsample=subsample)
    txt_pad, gls_pad = txt_vocab.pad, gls_vocab.pad
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate(b, txt_pad, gls_pad),
    )

    # Model
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    feat_size = cfg["data"]["feature_size"]
    if isinstance(feat_size, list):
        feat_size = sum(feat_size)
    model = build_model(
        cfg=cfg["model"],
        sgn_dim=feat_size,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    ).to(device)

    ckpt_path = model_dir / args.ckpt
    print(f"[ckpt] loading {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[ckpt] saved at epoch={state.get('epoch', '?')} "
          f"with bleu={state.get('best_bleu', '?')}")

    # Decode
    hyps, refs, names = [], [], []
    n_done = 0
    for batch in loader:
        batch = to_device(batch, device)
        out_ids = beam_search_decode(
            model, batch, txt_vocab,
            beam_size=args.beam_size,
            max_len=args.max_len,
            length_penalty=args.length_penalty,
        )
        for i in range(out_ids.shape[0]):
            hyp_words = ids_to_words(out_ids[i], txt_vocab)
            hyps.append(" ".join(hyp_words))
            refs.append(batch.ref_texts[i])
            names.append(batch.names[i])
        n_done += out_ids.shape[0]
        print(f"  decoded {n_done}/{len(items)}", end="\r")
    print()

    # Metrics
    bleu = compute_bleu(hyps, refs)
    wer = compute_wer(refs, hyps)
    chrf = compute_chrf(hyps, refs)
    rouge = compute_rouge_l(hyps, refs)
    print(f"\n=== beam={args.beam_size}, lp={args.length_penalty}, "
          f"max_len={args.max_len} ===")
    print(f"BLEU-4   : {bleu:.2f}")
    print(f"WER      : {wer:.2f}")
    print(f"chrF     : {chrf:.2f}")
    print(f"ROUGE-L  : {rouge:.2f}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            for n, r, h in zip(names, refs, hyps):
                f.write(f"{n}\t{r}\t{h}\n")
        print(f"[out] wrote per-sample (name, ref, hyp) to {out_p}")


if __name__ == "__main__":
    main()
