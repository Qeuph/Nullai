# Null AI

Null AI is a compact single-file language model trainer + inference stack centered on `null_ai.py`.

## Current training/data pipeline

- **Default dataset:** Databricks Dolly 15k (`databricks-dolly-15k.jsonl`).
- **Auto-download behavior:** If `--data` is not provided and local Dolly file is missing, the dataset is downloaded automatically.
- **Input format handling:**
  - JSONL with `instruction/context/response` fields is transformed into chat turns:
    `<|user|> ... <|assistant|> ...`
  - Plain text files are split on blank lines and wrapped into synthetic user/assistant pairs.
- **Train/val split:** 90/10 token split after tokenization.
- **Assistant-only supervision:** training masks loss outside `<|assistant|>` spans by default (`train_assistant_only=True`).
- **Progressive context curriculum:** default schedule grows train sequence length from 256 → 512 → 1024 → 2048 through training.

## Current tokenizer

Null AI currently uses `ChatTokenizer` (word/subword-like regex tokenizer), not byte-level BPE.

- **Special tokens:**
  - `<|pad|>` = 0
  - `<|bos|>` = 1
  - `<|eos|>` = 2
  - `<|unk|>` = 3
  - `<|user|>` = 4
  - `<|assistant|>` = 5
- **Vocabulary build:** top-frequency regex tokens from the loaded corpus, capped by `cfg.vocab_size` target (default 32k).
- **Saved vocab file:** `tokenizer_vocab.json`.
- **Runtime vocab:** `cfg.vocab_size` is overwritten with the trained tokenizer's actual vocabulary size.

## Model snapshot (project defaults)

- **Architecture:** 8 layers, `d_model=264`, GQA (`4Q / 2KV`), `d_mlp=1056`
- **Context length:** `max_seq_len=2048` (train sequence chunks default `seq_len=256`)
- **Core features:**
  - Partial RoPE
  - SmearGate
  - Sparse attention head-output gate
  - U-Net skip connections
  - Optional depth recurrence
  - Optional parallel decoder lane
  - SNN-inspired hypercube spike encoder
  - EMA + post-training int8 quantization

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python null_ai.py
```

Useful examples:

```bash
python null_ai.py --dataset dolly
python null_ai.py --data my_corpus.txt --max_iters 20000
python null_ai.py --d_model 384 --n_layers 10 --max_iters 15000
```

## Inference (interactive chat)

FP checkpoint:

```bash
python chat_loop.py --checkpoint null_ai_final.pt
```

Quantized checkpoint:

```bash
python chat_loop.py --checkpoint null_ai_final_int8.pt --quantized
```

### Decoding controls

- `--repetition_penalty` (default `1.1`)
- `--presence_penalty` (default `0.05`)
- `--frequency_penalty` (default `0.05`)
- `--repeat_window` (default `192`)
- `--no_repeat_ngram_size` (default `3`)
- `--min_new_tokens` (default `24`)
- Existing: `--temperature`, `--top_k`, `--top_p`, `--max_new_tokens`

The chat loop uses turn-structured history and trims full oldest turns to fit model context (`cfg.max_seq_len`) before generation.

### In-chat commands

- `/reset` — clear conversation history
- `/raw` — generate from a one-off prompt and print raw text
- `/quit` — exit

## PyTorch compatibility note

PyTorch 2.6+ changed `torch.load` default behavior (`weights_only=True`).
`chat_loop.py` includes compatibility loading for legacy checkpoints that store pickled config/classes.

## Validation checks

To quickly verify code health after updates:

```bash
python -m py_compile null_ai.py chat_loop.py
```

```bash
python chat_loop.py --help
```

> Note: running full training to completion is compute-intensive and not part of the lightweight sanity checks above.

## Files

- `null_ai.py` — model, tokenizer, data pipeline, training, quantization
- `chat_loop.py` — interactive inference client
- `requirements.txt` — Python dependencies
