# my-nanogpt

Blind-rewrite of [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) from scratch.

Part of my AI infra learning path. Rules:
- Watch Karpathy's videos as a reference, but **close the video and write the code myself**
- Only consult the video for a 2-3 min clip when stuck
- Everything verified against PyTorch's known behavior

## Progress

- [x] **Day 3 — `bigram.py`** — Bigram character LM on tiny Shakespeare
  - Char-level tokenizer (65 chars)
  - 1.1M-char Shakespeare dataset, 90/10 train/val split
  - Model: just an (V, V) embedding table, 4,225 params
  - Trained 3000 steps, loss 4.73 → 2.78
  - Autoregressive sampling with `torch.multinomial`
- [ ] Day 4 — single-head self-attention + positional embedding
- [ ] Day 5 — multi-head + FFN + residual + LayerNorm → full GPT

## Run

```bash
python bigram.py
```

Trains and generates a 500-char Shakespeare-ish sample. CPU-only, ~1 min on a laptop.

Expected generation (bigram ceiling, no real words):

```
sixJz$OMOTN at ha
TEG&$&Fr, bret cr dwDeNU-iD-s wojg ubs b d'CO:
```

## Dataset

[`input.txt`](input.txt) is the tiny Shakespeare corpus from
<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>.

## Notes

- [`bigram.ipynb`](bigram.ipynb) — exploratory notebook used during development
- [`bigram.py`](bigram.py) — clean, runnable artifact
- [`stuck.md`](stuck.md) — things I got stuck on and the fixes
