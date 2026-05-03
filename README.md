# Null AI

Null AI is a compact character-level language model trainer and inference stack implemented in a single file (`null_ai.py`). It targets T4-class GPUs and produces small checkpoints that are easy to run locally.

## Training run summary

The latest recorded run (provided in project notes) used:

- **Device:** CUDA
- **GPU:** Tesla T4
- **VRAM:** 15.6 GB
- **dtype:** bfloat16 mixed precision

### Model snapshot

- **Architecture:** 8 layers, `d_model=256`, GQA (`4Q / 2KV`), `d_mlp=1024`
- **Parameters:** 7,481,267
- **Model size (bf16):** 14.3 MB
- **Quantized checkpoint:** 7.3 MB (`null_ai_final_int8.pt`)
- **Best validation BPB:** 2.0384 (at step 4000)
- **Total steps:** 10,000
- **Total training time:** 53:55

## Features

- Grouped Query Attention (GQA)
- Partial RoPE + stabilized layernorm scaling
- SNN-inspired hypercube spike encoder
- Sparse attention gating
- U-Net skip connections
- Optional depth recurrence
- Parallel decoder lane
- TTT (test-time training) evaluation path
- EMA weights + int8 post-training quantization

## Requirements

- Python 3.10+
- PyTorch (CUDA build recommended for training)
- NumPy

Install basics:

```bash
pip install torch numpy
```

## Train

```bash
python null_ai.py
```

Useful options:

```bash
python null_ai.py --data my_corpus.txt --max_iters 20000
python null_ai.py --d_model 384 --n_layers 10 --max_iters 15000
```

## Inference chatbot loop

Use the interactive chatbot script:

```bash
python chat_loop.py --checkpoint null_ai_final.pt
```

or with the quantized checkpoint:

```bash
python chat_loop.py --checkpoint null_ai_final_int8.pt --quantized
```

### Commands inside chat

- `/reset` — clear conversation memory
- `/raw` — print raw generated text (including prompt prefix)
- `/quit` — exit

## PyTorch 2.6+ compatibility note

PyTorch 2.6 changed `torch.load` default behavior to `weights_only=True`, which can break loading checkpoints that contain pickled config/classes. `chat_loop.py` now loads full checkpoints in compatibility mode (`weights_only=False`, with fallback for older PyTorch) and registers legacy `__main__` symbols so older Null AI checkpoints still load.

If you trust your checkpoint source and still see loading errors, ensure you are launching with the updated `chat_loop.py` from this repository.

## Notes

- This is a **character-level** model, so outputs can drift into noisy Shakespeare-like text depending on temperature.
- If you trained on a different corpus, generation style will follow that corpus.
