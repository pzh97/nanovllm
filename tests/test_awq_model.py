"""End-to-end smoke test: an AWQ checkpoint must produce correct answers.

Run with: python -m tests.test_awq_model [model_path] [--cuda-graph]  (from the repo root, needs a GPU)
"""
import os
import sys
import time

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

QUESTIONS = [
    ("What is the capital of France? Answer with one word.", "paris"),
    ("What is 7 * 8? Answer with just the number.", "56"),
    ("Which planet is known as the Red Planet? Answer with one word.", "mars"),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = os.path.expanduser(args[0] if args else "~/huggingface/Qwen3-0.6B-AWQ")
    enforce_eager = "--cuda-graph" not in sys.argv
    tokenizer = AutoTokenizer.from_pretrained(path)
    t = time.perf_counter()
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=2048, max_num_batched_tokens=4096)
    print(f"engine ready in {time.perf_counter() - t:.1f}s")

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q, _ in QUESTIONS
    ]
    # temperature ~0 behaves like greedy decoding (exact greedy is not allowed by SamplingParams)
    outputs = llm.generate(prompts, SamplingParams(temperature=1e-5, max_tokens=32), use_tqdm=False)
    failed = 0
    for (q, expected), out in zip(QUESTIONS, outputs):
        ok = expected in out["text"].lower()
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q!r} -> {out['text']!r}")
    assert failed == 0, f"{failed} answers were wrong"
    print("AWQ model smoke test passed")


if __name__ == "__main__":
    main()
