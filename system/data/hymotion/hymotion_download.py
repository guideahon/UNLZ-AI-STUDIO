import argparse
import os
from pathlib import Path


def load_hf_token():
    try:
        from huggingface_hub import HfFolder
        return HfFolder.get_token()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    model_id = "tencent/HY-Motion-1.0"
    model_subdir = args.model.strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token = load_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    else:
        print("Warning: HF token not found. If the model is gated, download will fail.", flush=True)

    from huggingface_hub import snapshot_download

    print(f"Downloading {model_subdir} to {output_dir}", flush=True)
    snapshot_download(
        repo_id=model_id,
        token=token,
        local_dir=str(output_dir),
        allow_patterns=f"{model_subdir}/*",
    )
    print("Download completed.", flush=True)


if __name__ == "__main__":
    main()
