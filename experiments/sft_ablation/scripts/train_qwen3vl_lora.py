#!/usr/bin/env python3
"""LoRA SFT for the frozen Qwen3-VL ablation conditions."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


def read_rows(path: Path, video_dir: Path, limit: int | None) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            video = video_dir / Path(row["videos"][0]).name
            if not video.is_file() or video.stat().st_size == 0:
                continue
            row["_video_path"] = str(video)
            row["_source_line"] = line_number
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def structured_messages(row: dict[str, object], include_assistant: bool) -> list[dict[str, object]]:
    source_messages = row["messages"]
    user_text = str(source_messages[0]["content"]).replace("<video>\n", "", 1)
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": row["_video_path"],
                    "num_frames": 8,
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]
    if include_assistant:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": str(source_messages[1]["content"])}
                ],
            }
        )
    return messages


class JsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.rows[index]


class VideoSFTCollator:
    def __init__(self, processor: AutoProcessor, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        if len(rows) != 1:
            raise ValueError("SFT collator currently requires per-device batch size 1")
        row = rows[0]
        full = self.processor.apply_chat_template(
            structured_messages(row, include_assistant=True),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        prompt = self.processor.apply_chat_template(
            structured_messages(row, include_assistant=False),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        labels = full["input_ids"].clone()
        prompt_length = min(prompt["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_length] = -100
        labels[full["attention_mask"] == 0] = -100
        full["labels"] = labels
        return {key: value for key, value in full.items() if isinstance(value, torch.Tensor)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = read_rows(args.data, args.video_dir, args.limit)
    if not rows:
        raise ValueError("No available videos matched the training data")
    if args.limit is None:
        expected = sum(1 for _ in args.data.open(encoding="utf-8"))
        if len(rows) != expected:
            raise ValueError(f"Only {len(rows)}/{expected} training videos are available")
    random.Random(args.seed).shuffle(rows)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.video_processor.num_frames = 8
    processor.video_processor.max_frames = 8
    processor.video_processor.fps = None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    for parameter in model.visual.parameters():
        parameter.requires_grad = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=JsonlDataset(rows),
        data_collator=VideoSFTCollator(processor, args.max_length),
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    processor.save_pretrained(str(args.output_dir / "final_adapter"))
    metrics = {
        **result.metrics,
        "seed": args.seed,
        "data": str(args.data),
        "available_rows": len(rows),
        "num_frames": 8,
        "gradient_checkpointing": args.gradient_checkpointing,
        "trainable_scope": "language attention q/k/v/o LoRA; visual encoder frozen",
    }
    (args.output_dir / "run_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
