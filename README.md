# PHOENIX-2014T SLT Trainer

A self-contained trainer for a Sign Language Transformer (SLT) on the
RWTH-PHOENIX-Weather-2014T dataset, built on plain PyTorch (no torchtext
legacy, no signjoey). Useful for producing an independent back-translation
evaluator for sign-language production research.

## Layout

```
.
├── train_slt.py            # main trainer (single file)
├── config_seed142.yaml     # 4-layer / hidden=384 / seed=142 config
├── back_translation/       # transformer encoder / decoder modules
└── colab_train.ipynb       # ready-to-run Colab notebook
```

## Quick start (Colab)

Open `colab_train.ipynb` and follow the cells. You will need:

1. A Google Drive folder containing the three preprocessed pickles:
   - `phoenix14t.train.pickle.gzip`
   - `phoenix14t.dev.pickle.gzip`
   - `phoenix14t.test.pickle.gzip`
2. A T4 / A100 runtime (training takes ~3-6h on T4).

## Local (Linux / Windows with NVIDIA GPU)

```bash
pip install -r requirements.txt

python train_slt.py \
  --train slt_data/phoenix14t.train.pickle.gzip \
  --dev   slt_data/phoenix14t.dev.pickle.gzip \
  --test  slt_data/phoenix14t.test.pickle.gzip \
  --config config_seed142.yaml \
  --model_dir models/slt_seed142_4layer_h384 \
  --device cuda
```

## Notes on Mac (MPS)

`torch.ctc_loss` is **not implemented for the MPS backend** as of PyTorch 2.1.
You can either:

1. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for that op (slow), or
2. Use Colab / a CUDA machine (recommended).

## Resuming

The trainer writes `latest.ckpt` after every validation. Re-running the same
command resumes automatically (no flags needed).

## Output

`<model_dir>/best.ckpt` is the best dev-BLEU checkpoint. It can be loaded with
`back_translation.bt_model.build_model(...)` for downstream evaluation.
