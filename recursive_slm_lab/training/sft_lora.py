from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path

from ..memory import fetch_passed_episodes, register_adapter
from ..tasks import load_tasks
from ..loop.prompts import build_prompt


@dataclass
class TrainingResult:
    adapter_name: str
    adapter_path: str
    trained: bool
    message: str


def _resolve_target_modules(model) -> list[str]:
    import torch

    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    module_names = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    matched = [name for name in preferred if name in module_names]
    if matched:
        return matched
    return sorted(module_names)


def train_lora_adapter(conn, out_dir: str, base_model_path: str | None) -> TrainingResult:
    if not base_model_path:
        return TrainingResult("", out_dir, False, "RSLM_HF_MODEL_ID is not set.")

    required = ["torch", "datasets", "transformers", "peft"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        message = "Optional deps missing: datasets/peft/etc. Install: pip install -e '.[localhf,train]'"
        return TrainingResult("", out_dir, False, message)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    episodes = fetch_passed_episodes(conn)
    if not episodes:
        return TrainingResult("", out_dir, False, "No verified episodes to train on.")

    tasks = {task.task_id: task for task in load_tasks()}

    examples = []
    for ep in episodes:
        task = tasks.get(ep.task_id)
        if not task:
            continue
        prompt = build_prompt(task.prompt, task.function_name, task.signature, None, None)
        examples.append({"prompt": prompt, "code": ep.candidate_code.strip() + "\n"})

    if not examples:
        return TrainingResult("", out_dir, False, "No matching tasks found for training data.")

    dataset = Dataset.from_list(examples)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = "You are a code generation assistant."

    def tokenize(batch: dict) -> dict:
        prompt_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt_text + batch["code"]
        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=512, add_special_tokens=False)
        full_tokens = tokenizer(full_text, truncation=True, max_length=512, add_special_tokens=False)
        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()
        prompt_len = len(prompt_tokens["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    tokenized = dataset.map(tokenize, remove_columns=["prompt", "code"])

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    load_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = _resolve_target_modules(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def collate(features: list[dict]) -> dict:
        batch = tokenizer.pad(
            {"input_ids": [f["input_ids"] for f in features], "attention_mask": [f["attention_mask"] for f in features]},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            label = feature["labels"]
            pad_len = max_len - len(label)
            labels.append(label + [-100] * pad_len)
        batch["labels"] = torch.tensor(labels)
        return batch

    per_device_bs = 1
    grad_acc = 8
    steps_per_epoch = math.ceil(len(tokenized) / per_device_bs / grad_acc)
    max_steps = max(10, min(60, steps_per_epoch * 3))

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_acc,
        max_steps=max_steps,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collate)
    trainer.train()

    adapter_path = Path(out_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)

    adapter_name = adapter_path.name
    register_adapter(conn, adapter_name, str(adapter_path), notes="SFT LoRA adapter")

    return TrainingResult(adapter_name, str(adapter_path), True, "Adapter trained and registered.")
