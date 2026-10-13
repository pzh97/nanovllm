"""Closed-loop concurrency sweep: throughput, TTFT and TPOT vs concurrent users.

C users each keep exactly one request in flight; when a request finishes the
user immediately submits the next one. The same workload runs on nano-vllm or
vLLM's LLMEngine, so the numbers are directly comparable.

Usage (from the repo root):
    python -m benchmarks.sweep --model ~/huggingface/Qwen2.5-7B-Instruct-AWQ --out results/nanovllm_7b_awq.json
    python -m benchmarks.sweep --backend vllm --model ... --out results/vllm_7b_awq.json
"""
import argparse
import json
import os
import random
import statistics
import time


class NanoVLLMBackend:
    def __init__(self, args):
        from nanovllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            args.model,
            enforce_eager=args.enforce_eager,
            max_model_len=args.input_len + args.output_len + 16,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    def add(self, prompt_ids: list[int], output_len: int) -> int:
        sp = self.SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=output_len)
        return self.llm.add_request(prompt_ids, sp)

    def step(self) -> tuple[list[int], list[int]]:
        """Returns (ids that produced a token this step, ids that finished)."""
        outputs, _, new_tokens = self.llm.step()
        return [seq_id for seq_id, _ in new_tokens], [seq_id for seq_id, _, _ in outputs]


class VLLMBackend:
    def __init__(self, args):
        from vllm import EngineArgs, LLMEngine, SamplingParams
        self.SamplingParams = SamplingParams
        self.engine = LLMEngine.from_engine_args(EngineArgs(
            model=args.model,
            enforce_eager=args.enforce_eager,
            max_model_len=args.input_len + args.output_len + 16,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ))
        self.next_id = 0
        self.seen = {}

    def add(self, prompt_ids: list[int], output_len: int) -> int:
        rid = self.next_id
        self.next_id += 1
        sp = self.SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=output_len)
        self.engine.add_request(str(rid), {"prompt_token_ids": prompt_ids}, sp)
        self.seen[rid] = 0
        return rid

    def step(self):
        produced, finished = [], []
        for out in self.engine.step():
            rid = int(out.request_id)
            n = len(out.outputs[0].token_ids)
            if n > self.seen.get(rid, 0):
                self.seen[rid] = n
                produced.append(rid)
            if out.finished:
                finished.append(rid)
                self.seen.pop(rid, None)
        return produced, finished


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))]


def run_level(backend, concurrency, num_requests, input_len, output_len, rng):
    prompts = [[rng.randint(0, 10000) for _ in range(input_len)] for _ in range(num_requests)]
    t_add, t_first, t_last = {}, {}, {}
    submitted = 0

    def submit():
        nonlocal submitted
        rid = backend.add(prompts[submitted], output_len)
        t_add[rid] = time.perf_counter()
        submitted += 1

    start = time.perf_counter()
    for _ in range(min(concurrency, num_requests)):
        submit()
    done = 0
    while done < num_requests:
        produced, finished = backend.step()
        now = time.perf_counter()
        for rid in produced:
            t_first.setdefault(rid, now)
            t_last[rid] = now
        for rid in finished:
            done += 1
            if submitted < num_requests:
                submit()
    wall = time.perf_counter() - start

    ttft = [(t_first[r] - t_add[r]) * 1e3 for r in t_add]
    tpot = [(t_last[r] - t_first[r]) * 1e3 / (output_len - 1) for r in t_add]
    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "wall_s": wall,
        "output_throughput": num_requests * output_len / wall,
        "total_throughput": num_requests * (input_len + output_len) / wall,
        "ttft_ms_mean": statistics.mean(ttft),
        "ttft_ms_p50": pct(ttft, 50),
        "ttft_ms_p99": pct(ttft, 99),
        "tpot_ms_mean": statistics.mean(tpot),
        "tpot_ms_p99": pct(tpot, 99),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["nanovllm", "vllm"], default="nanovllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--requests-per-user", type=int, default=4)
    parser.add_argument("--min-requests", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.model = os.path.expanduser(args.model)
    assert max(args.concurrency) <= args.max_num_seqs

    backend = (NanoVLLMBackend if args.backend == "nanovllm" else VLLMBackend)(args)
    rng = random.Random(0)
    run_level(backend, 4, 4, args.input_len, 32, rng)  # warmup: compile kernels, fill caches

    results = []
    for c in args.concurrency:
        n = max(args.min_requests, c * args.requests_per_user)
        r = run_level(backend, c, n, args.input_len, args.output_len, rng)
        results.append(r)
        print(f"C={c:<3} out {r['output_throughput']:8.1f} tok/s | TTFT mean {r['ttft_ms_mean']:8.1f} ms "
              f"p99 {r['ttft_ms_p99']:8.1f} | TPOT mean {r['tpot_ms_mean']:6.1f} ms", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {k: v for k, v in vars(args).items() if k != "out"}
    meta["label"] = args.label or f"{args.backend} {os.path.basename(args.model.rstrip('/'))}"
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
