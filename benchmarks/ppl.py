"""WikiText-2 perplexity: how much quality does AWQ int4 cost?

AWQ checkpoints are dequantized with nanovllm's awq_dequantize (bit-identical
to what the Triton kernels compute) and evaluated with HF transformers, which
isolates the quantization error from the serving engine.

Usage (from the repo root):
    python -m benchmarks.ppl --models ~/huggingface/Qwen3-0.6B-official ~/huggingface/Qwen3-0.6B-AWQ
"""
import argparse
import glob
import math
import os

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from nanovllm.layers.quantization.awq import AWQConfig, awq_dequantize


def load_wikitext2_test() -> str:
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    return "".join(pq.read_table(path).column("text").to_pylist())


def load_model(path: str):
    config = AutoConfig.from_pretrained(path)
    quant = AWQConfig.from_hf(getattr(config, "quantization_config", None))
    if quant is None:
        return AutoModelForCausalLM.from_pretrained(path, dtype=config.dtype).cuda().eval(), "bf16" if config.dtype == torch.bfloat16 else "fp16"

    # Rebuild a dense checkpoint from the AWQ tensors.
    state = {}
    for file in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cuda") as f:
            for name in f.keys():
                if name.endswith((".qzeros", ".scales")):
                    continue
                if name.endswith(".qweight"):
                    prefix = name[: -len(".qweight")]
                    state[prefix + ".weight"] = awq_dequantize(
                        f.get_tensor(name), f.get_tensor(prefix + ".qzeros"), f.get_tensor(prefix + ".scales"),
                        quant.group_size, config.dtype,
                    )
                else:
                    state[name] = f.get_tensor(name).to(config.dtype)
    del config.quantization_config
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config, dtype=config.dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and all("lm_head" in k for k in missing), (missing, unexpected)
    if config.tie_word_embeddings:
        model.tie_weights()
    return model.eval(), f"AWQ int4 g{quant.group_size}"


@torch.inference_mode()
def perplexity(model, input_ids: torch.Tensor, seq_len: int) -> float:
    nll, count = 0.0, 0
    for start in range(0, input_ids.numel() - 1, seq_len):
        chunk = input_ids[start : start + seq_len + 1].cuda()
        if chunk.numel() < 2:
            break
        logits = model(chunk[None, :-1]).logits[0].float()
        nll += F.cross_entropy(logits, chunk[1:], reduction="sum").item()
        count += chunk.numel() - 1
    return math.exp(nll / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=0, help="evaluate only the first N tokens (0 = all)")
    args = parser.parse_args()

    text = load_wikitext2_test()
    rows = []
    for path in args.models:
        path = os.path.expanduser(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        input_ids = torch.tensor(tokenizer(text).input_ids)
        if args.max_tokens:
            input_ids = input_ids[: args.max_tokens]
        model, kind = load_model(path)
        weight_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
        ppl = perplexity(model, input_ids, args.seq_len)
        rows.append((os.path.basename(path.rstrip("/\\")), kind, ppl, input_ids.numel()))
        print(f"{rows[-1][0]:<32} {kind:<14} ppl={ppl:.3f}  ({input_ids.numel()} tokens, dense size {weight_gib:.2f} GiB)", flush=True)
        del model
        torch.cuda.empty_cache()

    print(f"\n| Model | Weights | WikiText-2 PPL (seq {args.seq_len}) |\n|---|---|---|")
    for name, kind, ppl, _ in rows:
        print(f"| {name} | {kind} | {ppl:.3f} |")


if __name__ == "__main__":
    main()
