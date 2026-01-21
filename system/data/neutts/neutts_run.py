import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from neutts import NeuTTS
    import soundfile as sf

    tts = NeuTTS(
        backbone_repo=args.backbone,
        backbone_device=args.device,
        codec_repo=args.codec,
        codec_device=args.device,
    )
    ref_codes = tts.encode_reference(args.ref_audio)
    wav = tts.infer(args.text, ref_codes, args.ref_text)
    sf.write(str(output_path), wav, 24000)
    print(str(output_path))


if __name__ == "__main__":
    main()
