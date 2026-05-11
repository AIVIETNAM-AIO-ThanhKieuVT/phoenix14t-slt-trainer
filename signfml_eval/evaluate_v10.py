#!/usr/bin/env python3
"""
Evaluate Model V10 and Baselines on all metrics.
Supports V4, V5/V8/V10 architectures and custom checkpoints.
"""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# Layout: this file lives at <repo_root>/signfml_eval/evaluate_v10.py
# - SCRIPT_DIR is signfml_eval/ — contains dataset_v4, models/, slt_wrapper_v5_28feb
# - REPO_ROOT is the trainer repo root — contains metrics.py, back_translation/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from dataset_v4 import Phoenix14TDatasetV4 as Phoenix14TDataset, collate_fn
from models.fml.autoencoder_v2 import UnifiedPoseAutoencoder
from slt_wrapper_v5_28feb import SLTWrapper

# Official metrics (from repo-root metrics.py copied from temp_slrtp_eval)
from metrics import bleu, chrf, rouge, wer as wer_func, pose_dtw_mje, pose_distance, pose_length, pose_acceleration
from fastdtw import fastdtw as dtw_fn


def pose_dtw_mje_per_seq(hyps, gt_pose):
    """Per-sequence DTW-MJE: average over sequences (not frames).

    The standard pose_dtw_mje concatenates all aligned frames and computes a
    single mean — this weights longer sequences more heavily. Generative models
    tend to produce longer sequences (duration ratio > 1), so they are penalized
    more than regression models. This function averages per-sequence MPJPE
    instead, giving equal weight to each sample regardless of length.
    """
    def euclidean_distance(x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        return torch.sqrt(torch.sum((x - y) ** 2))

    per_seq_errors = []
    for hyp, gt in zip(hyps, gt_pose):
        if hyp is None or gt is None or len(hyp) == 0 or len(gt) == 0:
            continue
        _, path = dtw_fn(hyp.flatten(1, -1), gt.flatten(1, -1), dist=euclidean_distance)
        a_idx, b_idx = zip(*path)
        aligned_hyp = hyp[list(a_idx)]   # (path_len, J, 3)
        aligned_gt  = gt[list(b_idx)]    # (path_len, J, 3)
        # per-joint error averaged over frames → scalar for this sequence
        seq_error = torch.mean(torch.norm(aligned_hyp - aligned_gt, dim=2)).item()
        per_seq_errors.append(seq_error)

    return float(np.mean(per_seq_errors)) if per_seq_errors else 0.0


def load_flow_model(model_version, ckpt_path, device):
    """Load the correct flow matcher model based on version."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if model_version == "v4":
        from models.fml.latent_flow_matcher_v4_27Feb import LatentFlowMatcherV4
        model = LatentFlowMatcherV4(
            latent_dim=256, hidden_dim=512,
            num_flow_layers=6, num_prior_layers=4, num_heads=8,
            max_seq_len=512, use_ssm_prior=True
        ).to(device)
    elif model_version == "v11":
        from models.fml.latent_flow_matcher_v11_24mar import LatentFlowMatcherV11
        cfg = ckpt.get('config', {})
        model = LatentFlowMatcherV11(
            latent_dim=cfg.get('latent_dim', 256),
            hidden_dim=cfg.get('hidden_dim', 512),
            num_fine_layers=cfg.get('num_fine_layers', 3),
            num_coarse_layers=cfg.get('num_coarse_layers', 3),
            num_prior_layers=cfg.get('num_prior_layers', 4),
            num_heads=cfg.get('num_heads', 8),
            max_seq_len=512, use_ssm_prior=cfg.get('use_ssm_prior', True),
            local_window=cfg.get('local_window', 64)
        ).to(device)
    elif model_version == "v11_1apr":
        from models.fml.latent_flow_matcher_v11_1apr import LatentFlowMatcherV11 as LatentFlowMatcherV11_1Apr
        cfg = ckpt.get('config', {})
        model = LatentFlowMatcherV11_1Apr(
            latent_dim=cfg.get('latent_dim', 256),
            hidden_dim=cfg.get('hidden_dim', 512),
            num_fine_layers=cfg.get('num_fine_layers', 3),
            num_coarse_layers=cfg.get('num_coarse_layers', 3),
            num_prior_layers=cfg.get('num_prior_layers', 4),
            num_heads=cfg.get('num_heads', 8),
            max_seq_len=512,
            use_ssm_prior=cfg.get('use_ssm_prior', True),
            cfg_dropout_rate=cfg.get('cfg_dropout_rate', 0.2),
            local_window=cfg.get('local_window', 64)
        ).to(device)
    elif model_version == "v11_single_scale":
        from models.fml.latent_flow_matcher_v11_single_scale import LatentFlowMatcherV11_SingleScale
        cfg = ckpt.get('config', {})
        model = LatentFlowMatcherV11_SingleScale(
            latent_dim=cfg.get('latent_dim', 256),
            hidden_dim=cfg.get('hidden_dim', 512),
            num_layers=cfg.get('num_layers', 9),
            num_heads=cfg.get('num_heads', 8),
            max_seq_len=512,
            use_ssm_prior=cfg.get('use_ssm_prior', True),
            local_window=cfg.get('local_window', 64)
        ).to(device)
    else:  # v5 / v8 / v10
        from models.fml.latent_flow_matcher_v5_28feb import LatentFlowMatcherV5
        model = LatentFlowMatcherV5(
            latent_dim=256, hidden_dim=512,
            num_flow_layers=6, num_prior_layers=4, num_heads=8,
            max_seq_len=512, use_ssm_prior=True
        ).to(device)

    msg = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.config = cfg # Attach config to model for convenience
    print(f"  Loaded {model_version.upper()}: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model, ckpt


def get_latent_stats(ckpt, device):
    """Extract latent mean/std from checkpoint."""
    lm_raw = ckpt.get('latent_mean', None)
    ls_raw = ckpt.get('latent_std', None)
    if lm_raw is None:
        cfg = ckpt.get('config', {})
        dim = cfg.get('latent_dim', 256)
        lm_raw = cfg.get('latent_mean', [0.0] * dim)
        ls_raw = cfg.get('latent_std', [1.0] * dim)
    if torch.is_tensor(lm_raw):
        return lm_raw.to(device).float(), ls_raw.to(device).float()
    return torch.tensor(lm_raw, dtype=torch.float32, device=device), \
           torch.tensor(ls_raw, dtype=torch.float32, device=device)


def run_eval(model_name, flow_matcher, ae, slt, dataloader, device, dataset_stats, ckpt,
             cfg_scale=2.5, temperature=1.0, steps=50, downsample_fps=True, smooth_sigma=0.0):
    mean = dataset_stats['mean'].to(device)
    std = dataset_stats['std'].to(device)
    lm, ls = get_latent_stats(ckpt, device)

    print(f"\n  [{model_name}] Latent: mean={lm.mean():.4f}, std={ls.mean():.4f}")
    print(f"  [{model_name}] cfg={cfg_scale}, temp={temperature}, steps={steps}")
    if downsample_fps:
        print(f"  [{model_name}] DTW eval at 12fps (downsample 25→12fps, matching official SLRTP toolkit)")
    if smooth_sigma > 0:
        print(f"  [{model_name}] Post-processing smoothing: Gaussian sigma={smooth_sigma}")

    # DTW: 12fps (downsample here). SLT: 25fps (wrapper does [::2] internally → 12fps)
    # This matches the official SLRTP toolkit where:
    #   - main.py downsamples to 12fps for DTW
    #   - back_translate() receives 12fps directly (no internal subsample)
    #   - Our wrapper already does [::2], so we feed 25fps to avoid double-downsample
    gt_all_poses, model_all_poses = [], []          # for DTW (12fps)
    gt_poses_slt, model_poses_slt = [], []          # for SLT (25fps, wrapper [::2] → 12fps)
    gt_texts, all_gt_hyp, all_model_hyp = [], [], []
    all_gt_lens, all_pred_lens = [], []
    
    # Flag to skip model if we only want GT stats
    skip_model = (flow_matcher is None)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{model_name}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            B = batch['poses'].shape[0]
            gt_texts.extend(batch['texts'])

            # GT poses
            gt_poses_norm = batch['poses']
            gt_lens = batch['seq_lengths']
            gt_poses_3d = (gt_poses_norm * std + mean).reshape(B, -1, 178, 3)
            for i in range(B):
                gt_seq_25fps = gt_poses_3d[i, :gt_lens[i]].cpu()
                gt_poses_slt.append(gt_seq_25fps)               # 25fps → wrapper [::2] → 12fps
                if downsample_fps:
                    gt_all_poses.append(gt_seq_25fps[::2])       # 12fps for DTW
                else:
                    gt_all_poses.append(gt_seq_25fps)
            all_gt_hyp.extend(slt.poses_to_text(gt_poses_slt[-B:]))

            if skip_model:
                continue

            # Model poses
            sample_out = flow_matcher.sample(
                batch, steps=steps, cfg_scale=cfg_scale, temperature=temperature, return_length=True
            )
            # V4 returns (latents, seq_lens), V5 returns (latents, x0, seq_lens)
            if len(sample_out) == 2:
                latents_std, pred_lens = sample_out
            else:
                latents_std, _, pred_lens = sample_out

            latents = latents_std * ls + lm
            
            if flow_matcher.config.get('latent_dim', 256) == 534: # Raw pose space
                model_poses_norm = latents
            else:
                model_poses_norm = ae.decode(latents)
                
            model_poses_3d = (model_poses_norm * std + mean).reshape(B, -1, 178, 3)
            for i in range(B):
                pred_seq_25fps = model_poses_3d[i, :pred_lens[i]].cpu()
                if smooth_sigma > 0:
                    pred_seq_25fps = torch.from_numpy(
                        gaussian_filter1d(pred_seq_25fps.numpy(), sigma=smooth_sigma, axis=0)
                    )
                model_poses_slt.append(pred_seq_25fps)               # 25fps → wrapper [::2] → 12fps
                if downsample_fps:
                    model_all_poses.append(pred_seq_25fps[::2])      # 12fps for DTW
                else:
                    model_all_poses.append(pred_seq_25fps)
            all_model_hyp.extend(slt.poses_to_text(model_poses_slt[-B:]))

            # Length analysis
            all_pred_lens.extend(pred_lens.cpu().tolist())

        # Length analysis
        all_gt_lens.extend(gt_lens.cpu().tolist())

    # Compute metrics
    print(f"\n  [{model_name}] Computing metrics...")

    gt_m = {
        "bleu": bleu(all_gt_hyp, gt_texts),
        "chrf": chrf(all_gt_hyp, gt_texts),
        "rouge": rouge(all_gt_hyp, gt_texts),
        "wer": wer_func(all_gt_hyp, gt_texts),
        "dtw": 0.0, 
        "dtw_per_seq": 0.0,
        "dist": 1.0, 
        "accel": 1.0, 
        "avg_dur": 1.0,
    }

    model_m = {}
    if not skip_model:
        model_m = {
            "bleu": bleu(all_model_hyp, gt_texts),
            "chrf": chrf(all_model_hyp, gt_texts),
            "rouge": rouge(all_model_hyp, gt_texts),
            "wer": wer_func(all_model_hyp, gt_texts),
            "dtw": pose_dtw_mje(hyps=model_all_poses, gt_pose=gt_all_poses),
            "dtw_per_seq": pose_dtw_mje_per_seq(model_all_poses, gt_all_poses),
            "dist": pose_distance(hyps=model_all_poses, gt_pose=gt_all_poses),
            "accel": pose_acceleration(hyps=model_all_poses, gt_pose=gt_all_poses),
            "avg_dur": pose_length(hyps=model_all_poses, gt_pose=gt_all_poses),
        }

        # Length analysis
        gt_lens_arr = np.array(all_gt_lens)
        pred_lens_arr = np.array(all_pred_lens)
        model_m["len_mse"] = np.mean((gt_lens_arr - pred_lens_arr) ** 2)
        model_m["len_mae"] = np.mean(np.abs(gt_lens_arr - pred_lens_arr))
        model_m["len_ratio"] = np.mean(pred_lens_arr / (gt_lens_arr + 1e-6))

    # NLTK BLEU for comparison with training metric
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    def _nltk_bleu(hyps, refs):
        return nltk_corpus_bleu([[r.lower().split()] for r in refs],
                                [h.lower().split() for h in hyps],
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smooth) * 100
    gt_m["nltk_bleu4"] = _nltk_bleu(all_gt_hyp, gt_texts)
    
    # sacrebleu for GT
    import sacrebleu
    gt_m["sacrebleu4"] = sacrebleu.corpus_bleu(
        [h.lower() for h in all_gt_hyp],
        [[r.lower() for r in gt_texts]],
        tokenize="none"
    ).score

    if not skip_model:
        model_m["nltk_bleu4"] = _nltk_bleu(all_model_hyp, gt_texts)
        model_m["sacrebleu4"] = sacrebleu.corpus_bleu(
            [h.lower() for h in all_model_hyp],
            [[r.lower() for r in gt_texts]],
            tokenize="none"
        ).score

    return gt_m, model_m


def print_table(results_dict, gt_m):
    header = f"{'Model':<20} | {'B1':>6} | {'B2':>6} | {'B3':>6} | {'B4':>6} | {'sacre':>6} | {'NLTK':>6} | {'CHRF':>6} | {'ROUGE':>6} | {'WER':>6} | {'MJE':>7} | {'MJE/seq':>8} | {'Dist':>6} | {'Accel':>6} | {'Dur':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    def row(name, m):
        b = m['bleu']
        sacre = m.get('sacrebleu4', '-')
        sacre_s = f"{sacre:>6.2f}" if isinstance(sacre, float) else f"{sacre:>6}"
        accel = m.get('accel', '-')
        accel_s = f"{accel:>6.3f}" if isinstance(accel, float) else f"{accel:>6}"
        dtw_ps = m.get('dtw_per_seq', '-')
        dtw_ps_s = f"{dtw_ps:>8.4f}" if isinstance(dtw_ps, float) else f"{dtw_ps:>8}"
        return (f"{name:<20} | {b['bleu1']:>6.2f} | {b['bleu2']:>6.2f} | {b['bleu3']:>6.2f} | {b['bleu4']:>6.2f} | "
                f"{sacre_s} | {m['nltk_bleu4']:>6.2f} | {m['chrf']:>6.2f} | {m['rouge']:>6.2f} | {m['wer']:>6.2f} | "
                f"{m['dtw']:>7.4f} | {dtw_ps_s} | {m['dist']:>6.4f} | {accel_s} | {m['avg_dur']:>6.3f}")

    print(row("Ground Truth", gt_m))
    for name, m in results_dict.items():
        if m: # Only print if not empty
            print(row(name, m))
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='V10')
    parser.add_argument('--model_version', type=str, default='v10', choices=['v4', 'v5', 'v8', 'v10', 'v11', 'v11_1apr'])
    parser.add_argument('--flow_ckpt', type=str, default=None)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--gt_only', action='store_true', help='Only evaluate ground truth stats')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--cfg_scale', type=float, default=2.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='evaluation_results/v10_eval')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default=None, help='Override data directory')
    parser.add_argument('--slt_model_dir', type=str, default=None, help='Override SLT model directory')
    parser.add_argument('--no_downsample', action='store_true',
                        help='Disable 25→12fps downsampling (NOT recommended: breaks comparability with official SLRTP toolkit)')
    parser.add_argument('--smooth_sigma', type=float, default=0.0,
                        help='Gaussian smoothing sigma applied to generated poses (0=disabled). Try 0.8, 1.2, 1.5, 2.0.')
    parser.add_argument('--output_json', type=str, help='Path to save results as JSON')
    args = parser.parse_args()

    # Set seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {args.seed}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    SLT_DIR = args.slt_model_dir or os.path.join(PROJECT_ROOT, "backTranslation_PHIX_model")
    DATA_DIR = args.data_dir or os.path.join(PROJECT_ROOT, "SignFML_DataExtraction_Phoenix14T/data")

    # Shared models
    print("Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(pose_dim=534, latent_dim=256).to(device)
    ae_sd = torch.load(args.ae_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in ae_sd:
        ae_sd = ae_sd['model_state_dict']
    ae.load_state_dict(ae_sd, strict=False)
    ae.eval()

    print("Loading SLT...")
    slt = SLTWrapper(model_dir=SLT_DIR)

    # Dataset
    dataset = Phoenix14TDataset(DATA_DIR, split=args.split, normalize=True)
    if args.num_samples:
        dataset.video_ids = dataset.video_ids[:args.num_samples]
    dataloader = DataLoader(dataset, batch_size=1 if device.type != 'cuda' else 16, shuffle=False, collate_fn=collate_fn)
    print(f"Split: {args.split}, Samples: {len(dataset)}")

    print(f"\n Evaluating {args.model_name}")
    flow_matcher, ckpt = None, {}
    if not args.gt_only:
        if args.flow_ckpt is None:
            print("Error: --flow_ckpt is required unless --gt_only is set.")
            return
        flow_matcher, ckpt = load_flow_model(args.model_version, args.flow_ckpt, device)

    gt_m, model_m = run_eval(
        args.model_name, flow_matcher, ae, slt, dataloader, device,
        dataset.stats, ckpt,
        cfg_scale=args.cfg_scale, temperature=args.temperature, steps=args.steps,
        downsample_fps=not args.no_downsample, smooth_sigma=args.smooth_sigma
    )

    results = {args.model_name: model_m} if not args.gt_only else {}
    print_table(results, gt_m)

    if args.output_json:
        full_results = []
        if gt_m:
            gt_row = {"Model": "Ground Truth"}
            gt_row.update(gt_m)
            full_results.append(gt_row)
        for name, m in results.items():
            row = {"Model": name}
            row.update(m)
            full_results.append(row)
        
        import json
        with open(args.output_json, 'w') as f:
            json.dump(full_results, f, indent=4)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
