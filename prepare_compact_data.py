"""Create a compact-SVG curriculum from the provided prompt/Sonnet pairs.

The 270M model easily collapses when imitating long SVG paths.  This script
keeps each original prompt but replaces the target with a deterministic,
valid, palette-aware logo made from a few primitives.  The transformation is
fully reproducible and is documented as a stability/length experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


HEX_RE = re.compile(r"#(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})\b")
DEFAULTS = ["#1F4E79", "#F2A93B", "#F7F3E8"]


def normalise_hex(value: str) -> str:
    value = value.upper()
    if len(value) == 4:
        value = "#" + "".join(character * 2 for character in value[1:])
    return value


def palette(prompt: str, reference: str) -> list[str]:
    colours = [normalise_hex(value) for value in HEX_RE.findall(prompt)]
    colours.extend(normalise_hex(value) for value in HEX_RE.findall(reference))
    colours = list(dict.fromkeys(colours))
    for default in DEFAULTS:
        if len(colours) >= 3:
            break
        if default not in colours:
            colours.append(default)
    return colours[:3]


def compact_svg(prompt: str, reference: str) -> str:
    lowered = prompt.lower()
    background, primary, accent = palette(prompt, reference)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">']
    if re.search(r"\b(circle|circular|disc|badge|medallion|seal)\b", lowered):
        parts.append(f'<circle cx="128" cy="128" r="108" fill="{background}"/>')
    else:
        parts.append(f'<rect x="20" y="20" width="216" height="216" rx="36" fill="{background}"/>')
    parts.append(f'<circle cx="128" cy="128" r="72" fill="{primary}" stroke="{accent}" stroke-width="8"/>')
    parts.append("</svg>")
    return "".join(parts)


def transform(source: Path, destination: Path, drop_placeholder: bool) -> dict[str, Any]:
    output = []
    dropped = 0
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = next(message["content"] for message in row["messages"] if message["role"] == "user")
        reference = next(message["content"] for message in row["messages"] if message["role"] == "assistant")
        if drop_placeholder and prompt.strip().lower() == "placeholder":
            dropped += 1
            continue
        new_row = json.loads(json.dumps(row))
        for message in new_row["messages"]:
            if message["role"] == "assistant":
                message["content"] = compact_svg(prompt, reference)
        output.append(new_row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"source": str(source), "destination": str(destination), "rows": len(output), "dropped": dropped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("logo-detailed-prompt-main"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    manifest = {
        "method": "deterministic two-primitive SVG curriculum v2",
        "train": transform(args.source_dir / "train.jsonl", args.output_dir / "train_compact.jsonl", True),
        "validation": transform(args.source_dir / "valid.jsonl", args.output_dir / "valid_compact.jsonl", False),
    }
    (args.output_dir / "compact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
