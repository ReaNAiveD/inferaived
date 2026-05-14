"""Dump HuggingFace Qwen 3.5 reference outputs for validating Rust kernels.

Saves per-layer hidden states (and finer-grained sub-module outputs) as
``.safetensors`` files so the Rust side can ``safetensors::SafeTensors`` them
back and compare numerically.

Usage examples (run from ``reference/``)::

    uv run dump_reference.py
    uv run dump_reference.py "Hello "
    uv run dump_reference.py "Hello world" "你好"
    uv run dump_reference.py --output-root outputs --model ../model/Qwen3.5-0.8B "Hello "
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default to the model already living next to the Rust workspace.
DEFAULT_MODEL_PATH = "../model/Qwen3.5-0.8B"
DEFAULT_PROMPTS = ["Hello "]


def prompt_slug(prompt: str) -> str:
    """Stable, filesystem-safe directory name for a prompt.

    Combines a short alphanumeric prefix (for human readability) with an
    8-char SHA-1 suffix (so unicode / punctuation prompts still get unique
    output directories).
    """
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", prompt).strip("_")[:30]
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{safe or 'empty'}_{digest}"


def find_text_decoder(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the transformer decoder stack regardless of wrapping.

    Handles three common layouts:

    * ``model.model.layers``                         (text-only ``ForCausalLM``)
    * ``model.model.language_model.layers``          (multimodal wrapper)
    * ``model.language_model.layers``                (no extra ``.model`` wrapper)
    """
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
        if hasattr(inner, "language_model"):
            candidates.append(inner.language_model)
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    for c in candidates:
        if hasattr(c, "layers") and hasattr(c, "norm"):
            return c
    raise RuntimeError(
        f"Cannot find decoder stack on model {type(model).__name__}; "
        "tried `.model`, `.model.language_model`, `.language_model`."
    )


def register_dump_hooks(decoder: torch.nn.Module, captures: dict[str, torch.Tensor]):
    """Register forward hooks for every interesting submodule.

    Captured tensor naming convention:
      * ``embed_output``                       — output of token embedding
      * ``layer_NN_self_attn_output``          — attn block output (full attention layers)
      * ``layer_NN_linear_attn_output``        — linear (gated delta) attn block output
      * ``layer_NN_mlp_output``                — MLP block output (pre-residual)
      * ``layer_NN_output``                    — full layer output (post both residuals)
      * ``final_norm_output``                  — model.norm output
    """

    def make_hook(name: str):
        def hook(_module, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            tensor = tensor.detach().to(torch.float32).cpu().contiguous()
            # Squeeze leading batch=1 dim so shapes match the Rust side, which
            # never carries a batch dimension internally.
            if tensor.ndim >= 1 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            captures[name] = tensor.contiguous()

        return hook

    handles = []
    if hasattr(decoder, "embed_tokens"):
        handles.append(
            decoder.embed_tokens.register_forward_hook(make_hook("embed_output"))
        )

    for i, layer in enumerate(decoder.layers):
        handles.append(layer.register_forward_hook(make_hook(f"layer_{i:02d}_output")))
        if hasattr(layer, "self_attn"):
            sa = layer.self_attn
            handles.append(
                sa.register_forward_hook(make_hook(f"layer_{i:02d}_self_attn_output"))
            )
            # Optional sub-module hooks (only present on full-attention layers).
            for sub_name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "q_norm",
                "k_norm",
                "o_proj",
            ):
                sub = getattr(sa, sub_name, None)
                if sub is not None:
                    handles.append(
                        sub.register_forward_hook(
                            make_hook(f"layer_{i:02d}_{sub_name}_output")
                        )
                    )
        if hasattr(layer, "linear_attn"):
            handles.append(
                layer.linear_attn.register_forward_hook(
                    make_hook(f"layer_{i:02d}_linear_attn_output")
                )
            )
        if hasattr(layer, "mlp"):
            handles.append(
                layer.mlp.register_forward_hook(make_hook(f"layer_{i:02d}_mlp_output"))
            )

    handles.append(decoder.norm.register_forward_hook(make_hook("final_norm_output")))
    return handles


def dump_one(
    prompt: str,
    model: torch.nn.Module,
    tokenizer,
    output_root: Path,
    add_special_tokens: bool = False,
) -> Path:
    captures: dict[str, torch.Tensor] = {}
    decoder = find_text_decoder(model)
    handles = register_dump_hooks(decoder, captures)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = enc.input_ids

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    captures["logits"] = (
        out.logits.detach().to(torch.float32).cpu().contiguous().squeeze(0)
    )

    slug = prompt_slug(prompt)
    out_dir = output_root / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    last_logits = out.logits[0, -1]
    top5 = torch.topk(last_logits, k=5)
    meta = {
        "prompt": prompt,
        "add_special_tokens": add_special_tokens,
        "token_ids": input_ids[0].tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        "next_token_top5": {
            "ids": top5.indices.tolist(),
            "logits": [float(v) for v in top5.values.tolist()],
            "tokens": tokenizer.convert_ids_to_tokens(top5.indices.tolist()),
        },
        "torch_dtype": "float32",
        "attn_implementation": "eager",
        "num_layers": len(decoder.layers),
        "tensor_names": sorted(captures.keys()),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for name, tensor in captures.items():
        save_file(
            {"data": tensor.contiguous()},
            str(out_dir / f"{name}.safetensors"),
        )

    print(f"  -> {out_dir} ({len(captures)} tensors)")
    print(
        "     top-5: "
        + ", ".join(
            f"{tok!r}={lg:.3f}"
            for tok, lg in zip(
                meta["next_token_top5"]["tokens"],
                meta["next_token_top5"]["logits"],
            )
        )
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump HuggingFace Qwen 3.5 reference outputs for kernel validation."
    )
    parser.add_argument(
        "prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="One or more prompts to dump (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to local HF model directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Where to write per-prompt subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Pass add_special_tokens=True to the tokenizer (default: off).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for inference (default: %(default)s).",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(
        f"Loading model from {args.model} "
        f"(dtype=float32, attn_implementation=eager, device={args.device})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model = model.to(args.device).eval()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for prompt in args.prompts:
        print(f"\nDumping prompt {prompt!r}...")
        dump_one(
            prompt,
            model,
            tokenizer,
            output_root,
            add_special_tokens=args.add_special_tokens,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
