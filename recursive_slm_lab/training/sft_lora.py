from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Iterable

from ..memory import fetch_passed_episodes, register_adapter


@dataclass
class TrainingResult:
    adapter_name: str
    adapter_path: str
    trained: bool
    message: str


def train_lora_adapter(conn, out_dir: str, base_model_path: str | None) -> TrainingResult:
    if not base_model_path:
        return TrainingResult("", out_dir, False, "RSLM_HF_MODEL_PATH is not set.")

    required = ["torch", "datasets", "transformers", "peft"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        return TrainingResult("", out_dir, False, f"Optional deps missing: {', '.join(missing)}")

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    episodes = fetch_passed_episodes(conn)
    if not episodes:
        return TrainingResult("", out_dir, False, "No verified episodes to train on.")

    texts = []
    for ep in episodes:
        text = f"{ep.prompt}\n\n{ep.candidate_code}"
        texts.append({"text": text})

    dataset = Dataset.from_list(texts)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        logging_steps=5,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized)
    trainer.train()

    adapter_path = Path(out_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)

    adapter_name = adapter_path.name
    register_adapter(conn, adapter_name, str(adapter_path), notes="SFT LoRA adapter")

    return TrainingResult(adapter_name, str(adapter_path), True, "Adapter trained and registered.")
