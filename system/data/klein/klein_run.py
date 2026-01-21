import argparse
import os
from pathlib import Path


def load_hf_token():
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def ensure_model_download(model_id, token):
    from huggingface_hub import HfApi, hf_hub_download

    print(f"Checking model cache: {model_id}", flush=True)
    api = HfApi()
    files = api.list_repo_files(model_id, token=token)
    total = len(files)
    for idx, filename in enumerate(files, start=1):
        print(f"Downloading {idx}/{total}: {filename}", flush=True)
        hf_hub_download(repo_id=model_id, filename=filename, token=token)
    print("Model files ready.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", default="")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args = parse_args()

    token = load_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    else:
        print("Warning: HF token not found. If the model is gated, download will fail.", flush=True)

    if args.download_only:
        ensure_model_download(args.model, token)
        print("Download completed.", flush=True)
        return

    if not args.prompt:
        raise SystemExit("Prompt is required.")

    import torch
    from diffusers import Flux2KleinPipeline

    device = resolve_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}", flush=True)

    ensure_model_download(args.model, token)
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype, token=token)
    pipe = pipe.to(device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    )
    image = result.images[0] if hasattr(result, "images") else result[0]

    output_path = Path(args.output or f"klein_{int(torch.randint(0, 999999, (1,)).item())}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved output to {output_path}", flush=True)


if __name__ == "__main__":
    main()
