#!/usr/bin/env python3
"""Interactive inference loop for Null AI checkpoints."""

import argparse
import sys
from collections import Counter
import torch

from null_ai import NullAI, NullAIConfig, ChatTokenizer, CharTokenizer, load_quantised


def _register_legacy_pickle_symbols() -> None:
    """Register legacy __main__ symbols used by older checkpoints."""
    main_mod = sys.modules["__main__"]
    setattr(main_mod, "NullAIConfig", NullAIConfig)
    setattr(main_mod, "NullAI", NullAI)
    setattr(main_mod, "CharTokenizer", CharTokenizer)


def _torch_load_compat(path: str, device: torch.device):
    """Compatibility loader for PyTorch >=2.6 weights_only changes."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older torch may not support weights_only argument.
        return torch.load(path, map_location=device)


def load_model(checkpoint: str, quantized: bool, device: torch.device):
    _register_legacy_pickle_symbols()
    if quantized:
        # Keep this path routed through null_ai.load_quantised so dequant logic stays centralized.
        # PyTorch >=2.6 defaults torch.load(weights_only=True); older quantized checkpoints may
        # pickle config classes, so force weights_only=False for this call path.
        original_torch_load = torch.load

        def _torch_load_quant_compat(path, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(path, *args, **kwargs)

        torch.load = _torch_load_quant_compat
        try:
            model, cfg = load_quantised(checkpoint, device)
        finally:
            torch.load = original_torch_load
    else:
        ckpt = _torch_load_compat(checkpoint, device)
        cfg = ckpt["cfg"]
        model = NullAI(cfg).to(device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser(description="Null AI interactive chatbot loop")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--quantized", action="store_true", help="Checkpoint is int8 quantized")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=180)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--presence_penalty", type=float, default=0.05)
    p.add_argument("--frequency_penalty", type=float, default=0.05)
    p.add_argument("--repeat_window", type=int, default=192)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--min_new_tokens", type=int, default=24)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, args.quantized, device)
    tokenizer = ChatTokenizer.load("tokenizer_vocab.json")

    history_turns = []

    print("Null AI chat loop ready. Type /quit to exit, /reset to clear history, /raw for raw output.")
    print(f"Using device: {device}")

    while True:
        user = input("\nYou: ").strip()

        if not user:
            continue
        if user == "/quit":
            print("Bye.")
            break
        if user == "/reset":
            history_turns = []
            print("History cleared.")
            continue

        raw_mode = False
        if user == "/raw":
            raw_mode = True
            user = input("Prompt: ").rstrip("\n")

        def _prompt_from_turns(turns, include_assistant_prefix=True):
            rendered = "".join(f"<|{r}|> {t}\n" for r, t in turns)
            if include_assistant_prefix:
                rendered += "<|assistant|>"
            return rendered

        def _encode_with_budget(turns, max_ctx, reserve_tokens=8):
            budget = max(32, max_ctx - reserve_tokens)
            trimmed = list(turns)
            while trimmed:
                prompt = _prompt_from_turns(trimmed, include_assistant_prefix=True)
                enc = tokenizer.encode(prompt, add_bos=True)
                if len(enc) <= budget:
                    return enc, trimmed
                trimmed = trimmed[1:]
            return tokenizer.encode("<|assistant|>", add_bos=True), []

        if not raw_mode:
            history_turns.append(("user", user))
            encoded, history_turns = _encode_with_budget(history_turns, cfg.max_seq_len, reserve_tokens=args.max_new_tokens + 8)
        else:
            encoded = tokenizer.encode(user, add_bos=True)
            if len(encoded) > cfg.max_seq_len - 1:
                encoded = [tokenizer.BOS] + encoded[-(cfg.max_seq_len - 1):]
        ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        print("Assistant: ", end="", flush=True)

        with torch.no_grad():
            # Implementing streaming manually for now by modifying generate or using a loop
            # To keep it simple, we'll use a modified generate-like loop here for streaming

            model.eval()
            max_new_tokens = args.max_new_tokens
            temperature = args.temperature
            top_k = args.top_k
            top_p = args.top_p
            repetition_penalty = args.repetition_penalty
            presence_penalty = args.presence_penalty
            frequency_penalty = args.frequency_penalty
            repeat_window = max(1, args.repeat_window)
            no_repeat_ngram_size = args.no_repeat_ngram_size
            min_new_tokens = args.min_new_tokens

            prompt_ids = ids
            kv_cache = None
            generated_text = ""

            for step in range(max_new_tokens):
                if step == 0:
                    logits, _, kv_cache = model(prompt_ids[:, -cfg.max_seq_len:], use_cache=True)
                else:
                    logits, _, kv_cache = model(prompt_ids[:, -1:], past_kv=kv_cache, use_cache=True)

                logits = logits[:, -1, :] / max(temperature, 1e-6)

                # Repetition/frequency/presence penalties (recency windowed)
                if repetition_penalty and repetition_penalty > 1.0:
                    special_exempt = {tokenizer.PAD, tokenizer.BOS, tokenizer.EOS, tokenizer.USER, tokenizer.ASSISTANT}
                    recent = prompt_ids[0, -repeat_window:].tolist()
                    counts = Counter(t for t in recent if t not in special_exempt)
                    for tok_id, count in counts.items():
                        if logits[0, tok_id] < 0:
                            logits[0, tok_id] *= repetition_penalty
                        else:
                            logits[0, tok_id] /= repetition_penalty
                        logits[0, tok_id] -= (presence_penalty + frequency_penalty * float(count))

                # No-repeat n-gram blocking
                if no_repeat_ngram_size and no_repeat_ngram_size > 1 and prompt_ids.size(1) >= no_repeat_ngram_size - 1:
                    generated = prompt_ids[0].tolist()
                    prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
                    banned = set()
                    for i in range(len(generated) - no_repeat_ngram_size + 1):
                        ng = generated[i:i + no_repeat_ngram_size]
                        if tuple(ng[:-1]) == prefix:
                            banned.add(ng[-1])
                    if banned:
                        logits[0, list(banned)] = float('-inf')

                # Top-k
                if top_k > 0:
                    k_val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < k_val[:, [-1]]] = float('-inf')

                # Top-p
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float('-inf')
                    logits.scatter_(1, sorted_idx, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                prompt_ids = torch.cat([prompt_ids, next_tok], dim=1)

                token_text = tokenizer.decode(next_tok[0].tolist(), skip_special=True)
                print(token_text, end="", flush=True)
                generated_text += token_text

                if step + 1 >= min_new_tokens and next_tok.item() == tokenizer.EOS:
                    break

                # Check for stop sequences in generated text
                if "\n<|user|>" in generated_text or "\n<|assistant|>" in generated_text:
                    break

            print() # End of line after generation

        if raw_mode:
            # For raw mode, we still printed it, but maybe the user wants something else.
            # In this implementation, streaming and raw mode both print to console.
            continue

        completion = generated_text.strip() or "..."
        stop_positions = [
            pos for pos in (
                completion.find("\n<|user|>"),
                completion.find("\n<|assistant|>"),
            ) if pos != -1
        ]
        if stop_positions:
            completion = completion[:min(stop_positions)]

        completion = completion.strip() or "..."
        print(f"Assistant: {completion}")

        history_turns.append(("assistant", completion))


if __name__ == "__main__":
    main()
