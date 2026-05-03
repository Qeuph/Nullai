#!/usr/bin/env python3
"""Interactive inference loop for Null AI checkpoints."""

import argparse
import sys
import torch

from null_ai import NullAI, NullAIConfig, CharTokenizer, load_quantised


def _register_legacy_pickle_symbols() -> None:
    """Register legacy __main__ symbols used by older checkpoints.

    Some checkpoints were saved while training scripts were executed directly,
    which pickles classes under ``__main__``. Expose those names here so
    ``torch.load(..., weights_only=False)`` can resolve them.
    """
    main_mod = sys.modules["__main__"]
    setattr(main_mod, "NullAIConfig", NullAIConfig)
    setattr(main_mod, "NullAI", NullAI)
    setattr(main_mod, "CharTokenizer", CharTokenizer)


_register_legacy_pickle_symbols()

def load_model(checkpoint: str, quantized: bool, device: torch.device):
    if quantized:
        model, cfg = load_quantised(checkpoint, device)
    else:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
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
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, args.quantized, device)
    tokenizer = CharTokenizer()

    history = ""

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
            history = ""
            print("History cleared.")
            continue

        raw_mode = False
        if user == "/raw":
            raw_mode = True
            user = input("Prompt: ").rstrip("\n")

        history += f"User: {user}\nAssistant:"
        ids = torch.tensor(tokenizer.encode(history, add_bos=True), dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        text = tokenizer.decode(out[0].tolist())

        if raw_mode:
            print(f"\nRAW:\n{text}")
            continue

        completion = text[len(history):]
        stop_positions = [
            pos for pos in (
                completion.find("\nUser:"),
                completion.find("\nAssistant:"),
            ) if pos != -1
        ]
        if stop_positions:
            completion = completion[:min(stop_positions)]

        completion = completion.strip() or "..."
        print(f"Assistant: {completion}")

        history += f" {completion}\n"


if __name__ == "__main__":
    main()
