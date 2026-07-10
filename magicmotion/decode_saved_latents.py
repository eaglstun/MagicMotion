"""Decode saved MagicMotion latents into a video, in a lean VAE-only process.

The full pipeline's final VAE decode is the single heaviest op and OOMs on a
memory-marginal (e.g. 64GB unified-memory) machine because the ~21GB of denoising
models still occupy RAM. Set MAGICMOTION_SAVE_LATENTS=<path> on the inference run to
persist the finished latents just before decode; if the decode then dies, recover the
video here — only the VAE (~1.6GB) is loaded, so there is ample room to decode.

Usage:
    python magicmotion/decode_saved_latents.py <latents.pt> <output.mp4> [fps]
    # force CPU if MPS still runs out: MAGICMOTION_DEVICE=cpu python ...
"""

import sys

import torch
from device_utils import get_device
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor


def main():
    latents_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mm_latents.pt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "samples/decoded.mp4"
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    device = get_device()
    # bf16 conv on CPU is emulated and extremely slow; use float32 there. GPUs keep bf16.
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"[ decode ] device={device} dtype={dtype}  latents={latents_path}  -> {output_path}")

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.enable_slicing()
    vae.enable_tiling()

    scaling_factor = getattr(vae.config, "scaling_factor", None) or 1.15258426
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)

    latents = torch.load(latents_path, map_location="cpu").to(device=device, dtype=dtype)
    # mirror MagicMotionPipeline.decode_latents
    latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    latents = 1 / scaling_factor * latents

    with torch.no_grad():
        frames = vae.decode(latents).sample

    video = video_processor.postprocess_video(video=frames, output_type="np")
    export_to_video(video[0], output_path, fps=fps)
    print(f"[ decode ] wrote {output_path}")


if __name__ == "__main__":
    main()
