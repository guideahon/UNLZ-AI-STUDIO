import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel


def _parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GLM-4.7-Flash with Unsloth + LoRA.")
    parser.add_argument("--dataset", required=True, help="Path to JSON/JSONL dataset.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--base-model", default="unsloth/GLM-4.7-Flash", help="Base HF model id.")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Max sequence length.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per device batch size.")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument("--export-gguf", action="store_true", help="Export merged model to GGUF.")
    parser.add_argument("--gguf-quant", default="q4_k_m", help="GGUF quantization method.")
    return parser.parse_args()


def _normalize_messages(example):
    if isinstance(example, dict):
        if "messages" in example and isinstance(example["messages"], list):
            return example["messages"]
        if "conversations" in example and isinstance(example["conversations"], list):
            messages = []
            for entry in example["conversations"]:
                role = entry.get("from", "")
                content = entry.get("value", "")
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                elif role == "system":
                    role = "system"
                else:
                    role = "user"
                messages.append({"role": role, "content": content})
            return messages
    return []


def main():
    args = _parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def to_text(example):
        if isinstance(example, dict) and isinstance(example.get("text"), str):
            return {"text": example["text"]}
        messages = _normalize_messages(example)
        if not messages:
            return {"text": ""}
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda row: isinstance(row.get("text"), str) and row["text"].strip() != "")

    if len(dataset) == 0:
        raise SystemExit("Dataset has no usable samples. Expecting messages/conversations/text fields.")

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        fp16=fp16,
        bf16=bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=training_args,
    )

    trainer.train()

    lora_dir = output_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    if args.export_gguf:
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=args.gguf_quant)
        except Exception as exc:
            print(f"[warn] GGUF export failed: {exc}")

    meta = {
        "base_model": args.base_model,
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
    }
    (output_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
