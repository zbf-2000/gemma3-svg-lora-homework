# logo_detailed_prompt — (detailed prompt → SVG logo) training pairs

Supervised pairs for teaching a small model to draw an SVG logo from a **detailed
visual prompt**. Each target SVG was produced by Claude Sonnet from that prompt.

## Files

| File | Rows | Contents |
|---|---|---|
| `train.jsonl` | 219 | training pairs |
| `valid.jsonl` | 17 | validation pairs |

Each line is one chat-format example:

```json
{"messages": [
  {"role": "system",    "content": "<SVG-designer instructions>"},
  {"role": "user",      "content": "<detailed visual prompt>"},
  {"role": "assistant", "content": "<complete <svg>…</svg>>"}
]}
```

- **Input** = the detailed prompt (`user`).
- **Target** = one complete `<svg …>…</svg>` document (`assistant`), `viewBox="0 0 256 256"`.
- Train loss should be masked to the assistant (SVG) tokens only.

## Provenance

- Built from 275 generated records. After dropping incompletes and repairing
  malformed SVGs (unbalanced `<g>`, duplicate `</svg>`), **253** valid
  detailed-prompt → Sonnet-SVG pairs remained; **17** are held out as a private
  test set (not included here), leaving **236** published.
- Raw-query augmentation rows are intentionally excluded — these are
  **detailed-prompt** pairs only.
