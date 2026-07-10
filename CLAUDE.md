# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MagicMotion is an **image-to-video generation** research project (ICCV 2025) that adds
trajectory control to a frozen CogVideoX backbone. Given an input image plus a trajectory
map, it animates objects along the defined path. Control is expressed at three densities,
one per training "stage":

- **stage1 — mask** (dense): per-frame segmentation masks
- **stage2 — box**: per-frame bounding boxes
- **stage3 — sparse_box**: boxes only on a few keyframes

The publicly released code here is **inference-only**. Training code (`magicmotion/train.py`,
`magicmotion/dataset`, `magicmotion/scripts/train`, `accelerate_configs`) is `.gitignore`d and
absent — `args.py` still carries all the training flags, but there is no trainer to consume them.

## Environment

```bash
conda env create -n magicmotion --file environment.yml
conda activate magicmotion
pip install git+https://github.com/huggingface/diffusers   # diffusers must come from git, not pip
# For trajectory construction (Grounded-SAM2):
pip install -e trajectory_construction/Grounded_SAM2
pip install --no-build-isolation -e trajectory_construction/Grounded_SAM2/grounding_dino
```

Pinned stack (see `environment.yml`): Python 3.10, torch 2.4.1 + CUDA 12.x, transformers
4.44.2, gradio 4.44.0. Runs on **CUDA, Apple Silicon (MPS), or CPU** — device is picked at
runtime by `magicmotion/device_utils.py` (`get_device()`: `MAGICMOTION_DEVICE` env override,
else CUDA > MPS > CPU). The `environment.yml` stack is still CUDA-pinned; on a Mac install a
matching torch/torchvision build for MPS instead.

## Model weights

Two sets of weights, both large and not in git:

- **Base model** — `THUDM/CogVideoX-5b-I2V`, auto-downloaded from HF on first run (VAE, T5
  text encoder + tokenizer, and the transformer weights that `MagicMotionTransformer3DModel`
  initializes from).
- **MagicMotion checkpoints** — download into `ckpts/` from `quanhaol/MagicMotion`:
  ```bash
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download quanhaol/MagicMotion --local-dir ckpts
  ```
  Layout: `ckpts/stage{1,2,3}/{mask,box,sparse_box}.pt` plus
  `ckpts/stage{2,3}/*_perception_head.pt`.

## Running inference

Always run from the **repo root** — the launch scripts do `python magicmotion/inference.py`
so `magicmotion/` lands on `sys.path` (that's why source uses flat imports like
`from args import get_args` and `from models.trajectory_controlnet import ...`), and asset/CSV
paths inside the demo CSVs are resolved relative to CWD (= repo root).

```bash
bash magicmotion/scripts/inference/inference_mask.sh        # stage1
bash magicmotion/scripts/inference/inference_box.sh         # stage2 (+ perception head)
bash magicmotion/scripts/inference/inference_sparse_box.sh  # stage3 (+ perception head)
bash magicmotion/scripts/app/app.sh                         # Gradio demo (app.py, stage2 weights)
```

**Gotcha:** `inference_box.sh` has `export CUDA_VISIBLE_DEVICES=1` uncommented — it targets the
_second_ GPU. On a single-GPU box, comment that line out (mask/sparse_box already have it
commented); on a Mac it's a no-op (no CUDA to hide). box + sparse_box pass
`--use_perception_head`; mask does not.

Inference is driven by a CSV (`--validation_args_csv`, demos in
`magicmotion/validation_args/demo/`), one video per row. Columns:
`validation_prompt, validation_images, validation_trajectory_maps, output_path,
controlnet_weights, seed`. Outputs land in `samples/` (gitignored).

**Memory / device placement** (in `inference.py`, driven by `device_utils.get_device()`).
CogVideoX-5b is fixed at 480x720x49 (it rejects other resolutions — learned positional
embeddings) and is **memory-marginal on a 64GB unified-memory Mac**. Strategy by env:

- **CUDA (default):** `enable_sequential_cpu_offload()` — fits ~23GB VRAM.
- **MPS / CPU (default):** `enable_model_cpu_offload()` — the reliable Mac path. Slow
  (~3 min/step: on unified memory "CPU offload" duplicates models into the _same_ RAM pool
  and swaps), but it completes where the resident path OOMs.
- `MAGICMOTION_NO_OFFLOAD=1`: everything resident (`pipe.to`) — fastest, but the VAE
  encode + denoising step overflow 64GB unless little else is running (close other apps).
- `MAGICMOTION_CPU_OFFLOAD=1`: force sequential offload (tightest memory; thrashes on MPS).
- `MAGICMOTION_STEPS=N`: denoising steps (default 50); fewer = faster/rougher.

Perf note: a full 50-step video is ~hours on a 64GB Mac; it's dramatically faster with
apps closed, or on 128GB (where `MAGICMOTION_NO_OFFLOAD=1` fits and runs resident).

**The VAE decode** (final step) is the heaviest op and often OOMs on MPS even when
denoising succeeds. Recover without re-running: set `MAGICMOTION_SAVE_LATENTS=<path>` to
persist the finished latents just before decode, then decode in a lean VAE-only process:
`python magicmotion/decode_saved_latents.py <latents.pt> <out.mp4> [fps]`
(add `MAGICMOTION_DEVICE=cpu` if MPS still OOMs — CPU has no watermark; it decodes in fp32).

## Architecture

The base CogVideoX-5b-I2V model is **frozen**; trajectory control is bolted on as a ControlNet
plus an auxiliary head. Three pieces cooperate:

1. **`MagicMotionTransformer3DModel`** (`models/magicmotion_transformer_3d.py`) — the CogVideoX
   denoising transformer, subclassed to optionally carry a **perception head**. When
   `use_perception_head=True`, intermediate block features are collected and fed to a
   `SemanticFPNHead` that predicts a segmentation mask (`models/latent_segmentation.py`, a
   small FPN of per-tensor conv stacks). This is the "latent segmentation" auxiliary signal
   used by the box/sparse_box stages; its weights load from `*_perception_head.pt`.

2. **`TrajectoryControlnet`** (`models/trajectory_controlnet.py`) — a parallel ControlNet that
   encodes the trajectory map into `trajectory_hidden_states` and concatenates them onto the
   transformer's hidden states (note the custom `InflatedConv3d` for 3D conv over the video
   latent). Config is derived from the transformer's own config (minus `use_perception_head`).
   Weights load from the per-stage `{mask,box,sparse_box}.pt`.

3. **`MagicMotionPipeline`** (`pipelines/pipeline_magicmotion.py`) — the diffusers-style I2V
   pipeline that wires VAE + T5 + transformer + controlnet + DPM scheduler and runs the
   denoising loop with trajectory guidance.

`models/magicmotion.py` is just a thin `nn.Module` pairing `transformer` + `controlnet`.

`inference.py` is the assembly point: it loads T5/tokenizer/VAE from the base model, builds the
transformer (loading perception-head weights if present), constructs the controlnet from the
transformer's config, loads controlnet weights, then builds and runs the pipeline over the CSV.

## Trajectory construction

To make your own trajectory maps instead of using the provided demos, use the scripts in
`trajectory_construction/` (backed by Grounded-SAM2, which must be `pip install -e`'d as above).
Each maps to a stage — see `trajectory_construction/README.md` for exact args:

- `plan_mask.py` — segments a mask trajectory from an **input video** (Grounded-SAM2)
- `plan_box.py` — segments the first frame from an **input image**, then propagates a box
- `plan_sparse_box.py` — like box, but keeps boxes only at `--sparse_box_index` frames

`first_frame_generation/` (t2i_flux.py, edit_image_flux.py) optionally generates/edits the input
image with FLUX; needs `pip install git+https://github.com/huggingface/image_gen_aux`.
