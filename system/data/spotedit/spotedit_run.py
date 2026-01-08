import argparse
import os
import sys
from pathlib import Path


def load_hf_token():
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def prepare_paths():
    app_root = Path(__file__).resolve().parents[3]
    backend_dir = app_root / "system" / "ai-backends" / "SpotEdit"
    return app_root, backend_dir


def ensure_model_download(model_id, token):
    try:
        from huggingface_hub import HfApi, hf_hub_download
        print(f"Checking model cache: {model_id}", flush=True)
        api = HfApi()
        files = api.list_repo_files(model_id, token=token)
        total = len(files)
        for idx, filename in enumerate(files, start=1):
            print(f"Downloading {idx}/{total}: {filename}", flush=True)
            hf_hub_download(repo_id=model_id, filename=filename, token=token)
        print("Model files ready.", flush=True)
    except Exception as exc:
        print(f"Model download error: {exc}", flush=True)
        raise


def build_pipeline(backend, token):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}", flush=True)

    if backend == "flux":
        from diffusers import FluxKontextPipeline
        from FLUX_kontext import generate, SpotEditConfig

        model_id = "black-forest-labs/FLUX.1-Kontext-dev"
        print(f"Loading model: {model_id}", flush=True)
        ensure_model_download(model_id, token)
        pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
        pipe = pipe.to(device)
        config = SpotEditConfig(threshold=0.2)
        return pipe, generate, config

    from diffusers import QwenImageEditPipeline
    from Qwen_image_edit import generate, SpotEditConfig

    model_id = "Qwen/Qwen-Image-Edit"
    print(f"Loading model: {model_id}", flush=True)
    ensure_model_download(model_id, token)
    pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
    pipe = pipe.to(device)
    config = SpotEditConfig(threshold=0.15)
    return pipe, generate, config


def blend_mask(original, edited, mask):
    import numpy as np
    from PIL import Image

    mask = mask.convert("L")
    mask_np = np.array(mask).astype("float32") / 255.0
    if mask_np.ndim == 2:
        mask_np = mask_np[:, :, None]

    orig_np = np.array(original).astype("float32")
    edit_np = np.array(edited).astype("float32")
    out = edit_np * mask_np + orig_np * (1.0 - mask_np)
    out = out.clip(0, 255).astype("uint8")
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["flux", "qwen"], default="qwen")
    parser.add_argument("--input", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    app_root, backend_dir = prepare_paths()
    sys.path.insert(0, str(backend_dir))

    token = load_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    else:
        print("Warning: HF token not found. If the model is gated, download will fail.", flush=True)

    from PIL import Image

    if args.download_only:
        model_id = "black-forest-labs/FLUX.1-Kontext-dev" if args.backend == "flux" else "Qwen/Qwen-Image-Edit"
        print(f"Downloading model only: {model_id}", flush=True)
        ensure_model_download(model_id, token)
        print("Download completed.", flush=True)
        return

    input_image = Image.open(args.input).convert("RGB")
    mask_image = Image.open(args.mask).convert("L")

    target_size = (1024, 1024)
    resized = input_image.resize(target_size, Image.LANCZOS) if input_image.size != target_size else input_image
    mask_resized = mask_image.resize(resized.size, Image.NEAREST)

    print("Building pipeline (first run may download weights)...", flush=True)
    pipe, generate, config = build_pipeline(args.backend, token)
    print("Running SpotEdit...", flush=True)
    result = generate(
        pipe,
        image=resized,
        prompt=args.prompt,
        config=config,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )

    edited = result.images[0] if hasattr(result, "images") else result[0]
    blended = blend_mask(resized, edited, mask_resized)
    if blended.size != input_image.size:
        blended = blended.resize(input_image.size, Image.LANCZOS)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
