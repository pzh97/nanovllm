"""Lesson 6 (capstone): does our from-scratch Qwen3 agree with nano-vllm's real engine?

nano-vllm runs the same weights through fused kernels, FlashAttention, a paged KV cache and bf16.
We run its engine, record the logits it computes at every step (a hook on lm_head), then replay
the exact same tokens through our model (teacher forcing) and compare step by step.

Run from the repo root:  python lessons/lesson6.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qwen3_scratch as scratch
from nanovllm import LLM, SamplingParams

STEPS = 32
PROMPT = "The capital of France is"

# --- Step 1: nano-vllm, recording the logits of every step ---
llm = LLM(scratch.MODEL_DIR, enforce_eager=True, gpu_memory_utilization=0.5)
engine_logits = []
llm.model_runner.model.lm_head.register_forward_hook(lambda m, i, o: engine_logits.append(o[0].float().clone()))
prompt_ids = llm.tokenizer.encode(PROMPT)
# Greedy is not allowed (temperature > 0), but a tiny temperature makes the sampler pick the argmax
out = llm.generate([prompt_ids], SamplingParams(temperature=1e-5, max_tokens=STEPS, ignore_eos=True), use_tqdm=False)[0]
engine_tokens, engine_logits = out["token_ids"], torch.stack(engine_logits)  # [STEPS, vocab]
print("nano-vllm :", repr(out["text"]))


def compare(name, ours):
    # cos: 1.0 = same direction; top1: does the same token win at each step?
    cos = torch.nn.functional.cosine_similarity(ours, engine_logits, dim=-1)
    top1 = (ours.argmax(-1) == engine_logits.argmax(-1)).float().mean().item()
    diff = (ours - engine_logits).abs().max().item()
    print(f"{name:14s} top-1 agree {top1 * 100:5.1f}%   min cosine {cos.min().item():.5f}   max |diff| {diff:.3f}")


for dtype in (torch.float32, torch.bfloat16):
    name = f"scratch {str(dtype)[6:]}"
    scratch.load(dtype)

    # --- Step 2: replay the engine's tokens through our model (teacher forcing) ---
    compare(name, scratch.step_logits(prompt_ids, engine_tokens))

    # --- Step 3: let our model generate on its own ---
    ours, _ = scratch.generate(prompt_ids, STEPS)
    same = next((i for i, (a, b) in enumerate(zip(ours, engine_tokens)) if a != b), STEPS)
    print(f"{'':14s} free-running: identical for the first {same} of {STEPS} tokens")
    if same < STEPS:
        # How close was the engine's own decision at the first disagreement?
        top2 = engine_logits[same].topk(2)
        words = [repr(llm.tokenizer.decode([t])) for t in top2.indices.tolist()]
        print(f"{'':14s} step {same}: engine's top-2 were {words[0]} vs {words[1]}, "
              f"logit gap only {(top2.values[0] - top2.values[1]).item():.3f}")
        print(f"{'':14s} ours:", repr(llm.tokenizer.decode(ours)))
