#!/usr/bin/env bash
# One-command stage1 (mask) render tuned for Apple Silicon / 64GB Mac.
#
# Bakes in everything we learned verifying the MPS path on 2026-07-10:
#   - uses the conda magicmotion env python (bare `python` is a pyenv 3.14 shim
#     with no einops, which kills the stock .sh scripts)
#   - denoises on MPS, saves latents just before the VAE decode
#   - if the MPS decode OOMs (it reliably does on 64GB), auto-falls back to a
#     lean CPU fp32 decode of the saved latents
#
# Usage:  bash run_mask_mps.sh [STEPS] [CSV]
#   STEPS  denoising steps (default 50; try 2 for a plumbing smoke test)
#   CSV    validation args csv (default: one-row mammoth demo)
#
# NOTE: use a ONE-ROW csv. MAGICMOTION_SAVE_LATENTS is a single file, so a multi-row
# csv (e.g. the stock demo_mask.csv, 2 rows) would overwrite earlier rows' latents and
# only the last row would be CPU-decodable. Render one video per invocation.
#
# Run from the repo root. Close other apps first — offload thrashes on unified memory.
set -euo pipefail
cd "$(dirname "$0")"

MMPY="/Users/eeaglstun/miniconda3/envs/magicmotion/bin/python"
STEPS="${1:-50}"
CSV="${2:-magicmotion/validation_args/demo/demo_mask_mammoth.csv}"
LATENTS="samples/latents_mask_$(basename "$CSV" .csv).pt"

[ -x "$MMPY" ] || { echo "!! magicmotion env python not found at $MMPY (conda env missing?)"; exit 1; }
"$MMPY" -c "import einops, torch; assert torch.backends.mps.is_available()" \
  || { echo "!! env sanity check failed (einops / MPS)"; exit 1; }

mkdir -p samples
echo ">> Denoising on MPS: STEPS=$STEPS  CSV=$CSV  (latents -> $LATENTS)"
MAGICMOTION_STEPS="$STEPS" MAGICMOTION_SAVE_LATENTS="$LATENTS" \
  "$MMPY" magicmotion/inference.py \
    --pretrained_model_name_or_path THUDM/CogVideoX-5b-I2V \
    --pretrained_controlnet_path ckpts/stage1/mask.pt \
    --validation_args_csv "$CSV" \
    --num_validation_videos 1 \
    --height 480 --width 720

# inference.py swallows a per-row decode OOM and still exits 0, so decode explicitly
# from the saved latents on CPU (fp32, no MPS watermark) to guarantee an mp4.
if [ -f "$LATENTS" ]; then
  OUT="samples/demo/stage1/$(basename "$CSV" .csv)_mps.mp4"
  echo ">> Decoding saved latents on CPU -> $OUT"
  MAGICMOTION_DEVICE=cpu "$MMPY" magicmotion/decode_saved_latents.py "$LATENTS" "$OUT" 8
  echo ">> Done: $OUT"
else
  echo "!! No latents at $LATENTS — denoise did not reach the save point. Check the log above."
  exit 1
fi
