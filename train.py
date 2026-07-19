"""LoRA fine-tuning for Gemma 3 270M with loss only on assistant SVG tokens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    # The published train set contains two unusable placeholder prompts.
    return [
        row for row in rows
        if next((m["content"].strip().lower() for m in row["messages"] if m["role"] == "user"), "")
        != "placeholder"
    ]


class SvgDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        overlength_policy: str,
        split_name: str,
        max_samples: int | None = None,
    ):
        self.items = []
        self.dropped_overlength: list[int] = []
        self.truncated_overlength: list[int] = []
        self.excluded_by_sample_limit: list[int] = []
        for source_index, row in enumerate(rows):
            messages = row["messages"]
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(f"{split_name} row {source_index} does not end with an assistant target.")
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            encoded = tokenizer(full_text, add_special_tokens=False)
            input_ids = encoded["input_ids"]
            if len(input_ids) > max_length:
                if overlength_policy == "drop":
                    self.dropped_overlength.append(source_index)
                    continue
                if overlength_policy == "error":
                    raise ValueError(
                        f"{split_name} row {source_index} has {len(input_ids)} tokens, "
                        f"exceeding max_length={max_length}."
                    )
                if overlength_policy != "truncate":
                    raise ValueError(f"Unknown overlength policy: {overlength_policy}")
                self.truncated_overlength.append(source_index)
                input_ids = input_ids[:max_length]
                encoded["attention_mask"] = encoded["attention_mask"][:max_length]
            prefix = 0
            for full_id, prompt_id in zip(input_ids, prompt_ids):
                if full_id != prompt_id:
                    break
                prefix += 1
            if prefix != len(prompt_ids):
                raise ValueError(f"Chat-template prefix mismatch in {split_name} row {source_index}.")
            labels = list(input_ids)
            labels[:prefix] = [-100] * prefix
            if any(label != -100 for label in labels):
                self.items.append({
                    "input_ids": input_ids,
                    "attention_mask": encoded["attention_mask"],
                    "labels": labels,
                    "source_index": source_index,
                })
        if max_samples is not None and len(self.items) > max_samples:
            self.items.sort(key=lambda item: len(item["input_ids"]))
            self.excluded_by_sample_limit = [int(item["source_index"]) for item in self.items[max_samples:]]
            self.items = self.items[:max_samples]
        for item in self.items:
            item.pop("source_index", None)
        if not self.items:
            raise ValueError("No usable training examples after tokenisation.")
        print(
            f"{split_name}: kept={len(self.items)}, dropped_overlength={len(self.dropped_overlength)}, "
            f"excluded_by_sample_limit={len(self.excluded_by_sample_limit)}, "
            f"truncated_overlength={len(self.truncated_overlength)}, max_length={max_length}"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class Collator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        length = math.ceil(max(len(x["input_ids"]) for x in features) / 8) * 8
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = length - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class ChunkedLinearCrossEntropy(torch.autograd.Function):
    """Cross entropy without retaining [sequence, vocabulary] logits.

    Gemma 3 uses a very large vocabulary. On a 6 GB GPU, ordinary causal-LM
    loss materialises enough logits to exhaust memory even though the 270M
    model itself is small. The LM head is frozen during LoRA training, so the
    backward pass only needs to reconstruct gradients for hidden states.
    """

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk_size: int) -> torch.Tensor:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        valid_count = (flat_labels != -100).sum()
        if valid_count.item() == 0:
            raise ValueError("Batch contains no assistant tokens for the loss.")
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, flat_hidden.shape[0], chunk_size):
                end = min(start + chunk_size, flat_hidden.shape[0])
                logits = F.linear(flat_hidden[start:end], weight)
                total += F.cross_entropy(
                    logits.float(), flat_labels[start:end], ignore_index=-100, reduction="sum"
                )
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk_size = chunk_size
        ctx.valid_count = valid_count
        return total / valid_count

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        hidden, weight, labels = ctx.saved_tensors
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        grad_hidden = torch.empty_like(flat_hidden)
        scale = grad_output.float() / ctx.valid_count

        for start in range(0, flat_hidden.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, flat_hidden.shape[0])
            labels_chunk = flat_labels[start:end]
            logits = F.linear(flat_hidden[start:end].detach(), weight)
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            valid = labels_chunk != -100
            if valid.any():
                row_ids = torch.arange(end - start, device=hidden.device)[valid]
                probabilities[row_ids, labels_chunk[valid]] -= 1.0
            probabilities[~valid] = 0.0
            probabilities.mul_(scale)
            grad_hidden[start:end] = F.linear(probabilities.to(weight.dtype), weight.t())
        return grad_hidden.reshape_as(hidden), None, None, None


class MemoryEfficientTrainer(Trainer):
    def __init__(self, *args: Any, loss_chunk_size: int = 32, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss_chunk_size = loss_chunk_size

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        labels = inputs["labels"]
        causal_model = model.get_base_model()
        backbone = getattr(causal_model, causal_model.base_model_prefix)
        outputs = backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        # Standard causal-language-model shift: position t predicts token t+1.
        hidden = outputs.last_hidden_state[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        output_weight = causal_model.get_output_embeddings().weight
        loss = ChunkedLinearCrossEntropy.apply(
            hidden, output_weight, shifted_labels, self.loss_chunk_size
        )
        # Transformers 5.x passes the number of supervised causal tokens over
        # the whole gradient-accumulation window. Convert each microbatch mean
        # into its contribution to that global token mean.
        if num_items_in_batch is not None:
            local_items = shifted_labels.ne(-100).sum().to(loss.device)
            denominator = torch.as_tensor(num_items_in_batch, device=loss.device, dtype=loss.dtype)
            loss = loss * local_items.to(loss.dtype) / denominator
        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional short smoke test.")
    parser.add_argument("--checkpoint-dir", help="Override the Trainer checkpoint directory.")
    parser.add_argument("--adapter-dir", help="Override the final adapter directory.")
    parser.add_argument("--metrics-file", help="Override the training metrics JSON path.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])
    checkpoint_dir = args.checkpoint_dir or config["checkpoint_dir"]
    adapter_dir = args.adapter_dir or config["adapter_dir"]
    metrics_file = args.metrics_file or config["training_metrics_file"]

    model_path = config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data = SvgDataset(
        load_rows(config["train_file"]), tokenizer, config["train_max_length"],
        config["long_example_policy"], "train", config.get("max_train_samples"),
    )
    valid_data = SvgDataset(
        load_rows(config["validation_file"]), tokenizer, config["validation_max_length"],
        "error", "validation", None,
    )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    # SDPA and a shorter sequence length are important on 6 GB GPUs. Gemma 3
    # has a large vocabulary, so materialising training logits is expensive.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    training = config["training"]
    arguments = TrainingArguments(
        output_dir=checkpoint_dir,
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training["gradient_accumulation"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_steps"],
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=training["eval_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config["dtype"] == "bfloat16",
        fp16=config["dtype"] == "float16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        prediction_loss_only=True,
        remove_unused_columns=False,
        seed=config["seed"],
        max_steps=args.max_steps,
    )
    trainer = MemoryEfficientTrainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=Collator(tokenizer),
        processing_class=tokenizer,
        loss_chunk_size=32,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_config_path = Path(adapter_dir, "adapter_config.json")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = config["base_model_id"]
    adapter_config_path.write_text(json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics = {
        **result.metrics,
        **trainer.evaluate(),
        "train_rows": len(train_data),
        "valid_rows": len(valid_data),
        "train_dropped_overlength": len(train_data.dropped_overlength),
        "train_dropped_overlength_indices_after_placeholder_filter": train_data.dropped_overlength,
        "train_excluded_by_sample_limit": len(train_data.excluded_by_sample_limit),
        "train_excluded_by_sample_limit_indices_after_placeholder_filter": train_data.excluded_by_sample_limit,
        "validation_dropped_overlength": len(valid_data.dropped_overlength),
    }
    Path(metrics_file).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
