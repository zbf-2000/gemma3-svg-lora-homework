"""Resumable, fixed-setting base-vs-LoRA evaluation for the SVG assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import accelerate
import peft
import torch
import transformers
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from reward import REWARD_VERSION, SECTION_MAX, extract_svg, score_svg


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_model(path: str, dtype: torch.dtype) -> Any:
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model


def stop_token_ids(tokenizer: Any) -> tuple[list[int], int | None]:
    ids = [tokenizer.eos_token_id]
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if (
        isinstance(end_of_turn, int)
        and end_of_turn >= 0
        and end_of_turn != tokenizer.unk_token_id
        and end_of_turn not in ids
    ):
        ids.append(end_of_turn)
    return ids, end_of_turn if end_of_turn in ids else None


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    settings: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    stop_ids, end_of_turn_id = stop_token_ids(tokenizer)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(settings["repetition_penalty"]),
            no_repeat_ngram_size=int(settings.get("no_repeat_ngram_size", 0)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_ids,
            stop_strings=["</svg>"],
            tokenizer=tokenizer,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    token_ids = new_tokens.tolist()
    last_token = token_ids[-1] if token_ids else None
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if last_token == end_of_turn_id:
        finish_reason = "end_of_turn"
    elif last_token == tokenizer.eos_token_id:
        finish_reason = "eos"
    elif len(token_ids) >= int(settings["max_new_tokens"]):
        finish_reason = "length"
    elif re.search(r"</svg\s*>\s*$", text, re.I):
        finish_reason = "svg_close"
    else:
        finish_reason = "other"
    return text, {
        "new_tokens": len(token_ids),
        "finish_reason": finish_reason,
        "hit_max_new_tokens": finish_reason == "length",
    }


def run_pending(
    label: str,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
    existing: list[dict[str, Any]],
    save_progress: Any,
) -> list[dict[str, Any]]:
    records = {int(record["index"]): record for record in existing}
    for index, row in enumerate(rows):
        if index in records:
            print(f"{label} {index + 1}/{len(rows)}: resumed")
            continue
        messages = [message for message in row["messages"] if message["role"] != "assistant"]
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        output, generation = generate(model, tokenizer, messages, settings)
        scores = score_svg(prompt, output)
        records[index] = {"index": index, "output": output, "generation": generation, "scores": scores}
        ordered = [records[key] for key in sorted(records)]
        save_progress(label, ordered)
        print(
            f"{label} {index + 1}/{len(rows)}: reward={scores['total']:.2f}, "
            f"valid={scores['is_valid']}, stop={generation['finish_reason']}, tokens={generation['new_tokens']}"
        )
    return [records[key] for key in sorted(records)]


def summarise(records: list[dict[str, Any]], train_targets: set[str]) -> dict[str, Any]:
    totals = [float(record["scores"]["total"]) for record in records]
    extracted = [extract_svg(record["output"]) for record in records]
    complete = [svg for svg in extracted if svg is not None]
    output_hashes = [hashlib.sha256(record["output"].encode("utf-8")).hexdigest() for record in records]
    complete_hashes = [hashlib.sha256(svg.encode("utf-8")).hexdigest() for svg in complete]
    finish_reasons = Counter(record["generation"]["finish_reason"] for record in records)
    return {
        "mean_reward": round(statistics.mean(totals), 3),
        "median_reward": round(statistics.median(totals), 3),
        "reward_stddev": round(statistics.pstdev(totals), 3),
        "min_reward": round(min(totals), 3),
        "max_reward": round(max(totals), 3),
        "valid_svg_rate": round(statistics.mean(bool(record["scores"]["is_valid"]) for record in records), 5),
        "complete_svg_rate": round(len(complete) / len(records), 5),
        "output_only_svg_rate": round(
            statistics.mean(bool(record["scores"]["details"].get("output_only_svg", False)) for record in records), 5
        ),
        "xml_parseable_rate": round(
            statistics.mean(bool(record["scores"]["details"].get("xml_parseable", False)) for record in records), 5
        ),
        "length_limit_rate": round(
            statistics.mean(bool(record["generation"]["hit_max_new_tokens"]) for record in records), 5
        ),
        "mean_new_tokens": round(statistics.mean(record["generation"]["new_tokens"] for record in records), 1),
        "finish_reasons": dict(finish_reasons),
        "unique_output_rate": round(len(set(output_hashes)) / len(records), 5),
        "unique_complete_over_all_rate": round(len(set(complete_hashes)) / len(records), 5),
        "exact_train_target_matches": sum(svg in train_targets for svg in complete),
        "section_means": {
            name: round(statistics.mean(record["scores"]["sections"][name] for record in records), 3)
            for name in SECTION_MAX
        },
    }


def evaluation_signature(config: dict[str, Any], selected_indices: list[int]) -> tuple[str, dict[str, Any]]:
    adapter_file = Path(config["adapter_dir"], "adapter_model.safetensors")
    payload = {
        "seed": config["seed"],
        "base_model_id": config["base_model_id"],
        "model_name_or_path": config["model_name_or_path"],
        "adapter_sha256": sha256_file(adapter_file),
        "validation_sha256": sha256_file(config["evaluation_file"]),
        "reward_sha256": sha256_file(Path(__file__).with_name("reward.py")),
        "evaluator_sha256": sha256_file(__file__),
        "reward_version": REWARD_VERSION,
        "generation": config["generation"],
        "selected_validation_indices": selected_indices,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_config.yaml")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--restart", action="store_true", help="Discard an incompatible/interrupted partial run.")
    parser.add_argument("--max-samples", type=int, help="Pilot only: evaluate the first N validation rows.")
    parser.add_argument("--sample-indices", help="Pilot only: comma-separated validation row indices.")
    parser.add_argument("--adapter-dir", help="Pilot only: override adapter/checkpoint path.")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.adapter_dir:
        config["adapter_dir"] = args.adapter_dir
    output_path = Path(args.output)
    partial_path = output_path.with_name(output_path.name + ".partial")
    all_rows = load_rows(config["evaluation_file"])
    if args.sample_indices:
        if args.max_samples is not None:
            raise ValueError("Use either --sample-indices or --max-samples, not both")
        selected_indices = [int(value.strip()) for value in args.sample_indices.split(",") if value.strip()]
        if not selected_indices or any(index < 0 or index >= len(all_rows) for index in selected_indices):
            raise ValueError("Invalid --sample-indices")
    elif args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        selected_indices = list(range(min(args.max_samples, len(all_rows))))
    else:
        selected_indices = list(range(len(all_rows)))
    signature, signature_payload = evaluation_signature(config, selected_indices)

    if args.restart and partial_path.exists():
        partial_path.unlink()
    if partial_path.exists():
        progress = json.loads(partial_path.read_text(encoding="utf-8"))
        if progress.get("signature") != signature:
            raise RuntimeError(
                f"Partial evaluation {partial_path} was produced by different code/config/weights. "
                "Run again with --restart."
            )
        print(f"Resuming evaluation from {partial_path}")
    else:
        progress = {"signature": signature, "signature_payload": signature_payload, "base": [], "adapter": []}
        atomic_json_write(partial_path, progress)

    set_seed(int(config["seed"]))
    rows = [all_rows[index] for index in selected_indices]
    train_targets = set()
    for row in load_rows(config["train_file"]):
        user_text = next(message["content"].strip() for message in row["messages"] if message["role"] == "user")
        if user_text.lower() != "placeholder":
            train_targets.add(
                next(message["content"].strip() for message in row["messages"] if message["role"] == "assistant")
            )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def save_progress(label: str, records: list[dict[str, Any]]) -> None:
        progress[label] = records
        atomic_json_write(partial_path, progress)

    if len(progress["base"]) < len(rows):
        base = load_model(config["model_name_or_path"], dtype)
        progress["base"] = run_pending(
            "base", base, tokenizer, rows, config["generation"], progress["base"], save_progress
        )
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(progress["adapter"]) < len(rows):
        adapted_base = load_model(config["model_name_or_path"], dtype)
        adapted = PeftModel.from_pretrained(adapted_base, config["adapter_dir"])
        adapted.eval()
        progress["adapter"] = run_pending(
            "adapter", adapted, tokenizer, rows, config["generation"], progress["adapter"], save_progress
        )
        del adapted, adapted_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_records = progress["base"]
    adapter_records = progress["adapter"]
    reference_records = []
    samples = []
    for index, row in enumerate(rows):
        prompt = next(message["content"] for message in row["messages"] if message["role"] == "user")
        reference_svg = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
        reference_record = {
            "index": index,
            "output": reference_svg,
            "generation": {"new_tokens": 0, "finish_reason": "reference", "hit_max_new_tokens": False},
            "scores": score_svg(prompt, reference_svg),
        }
        reference_records.append(reference_record)
        samples.append({
            "index": index,
            "prompt": prompt,
            "reference_svg": reference_svg,
            "reference_scores": reference_record["scores"],
            "base": base_records[index],
            "adapter": adapter_records[index],
        })

    base_summary = summarise(base_records, train_targets)
    adapter_summary = summarise(adapter_records, train_targets)
    reference_summary = summarise(reference_records, train_targets)
    delta = {
        "mean_reward": round(adapter_summary["mean_reward"] - base_summary["mean_reward"], 3),
        "valid_svg_rate": round(adapter_summary["valid_svg_rate"] - base_summary["valid_svg_rate"], 5),
        "complete_svg_rate": round(adapter_summary["complete_svg_rate"] - base_summary["complete_svg_rate"], 5),
        "output_only_svg_rate": round(adapter_summary["output_only_svg_rate"] - base_summary["output_only_svg_rate"], 5),
        "length_limit_rate": round(adapter_summary["length_limit_rate"] - base_summary["length_limit_rate"], 5),
        "section_means": {
            name: round(adapter_summary["section_means"][name] - base_summary["section_means"][name], 3)
            for name in SECTION_MAX
        },
    }
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_signature": signature,
        "evaluation_inputs": signature_payload,
        "environment": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "accelerate": accelerate.__version__,
        },
        "config": config,
        "validation_examples": len(rows),
        "selected_validation_indices": selected_indices,
        "summary": {"base": base_summary, "adapter": adapter_summary, "reference": reference_summary, "delta": delta},
        "samples": samples,
    }
    atomic_json_write(output_path, result)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
