import os
from nanovllm import LLM, SamplingParams

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

llm = LLM(
    path,
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=10,
)

all_token_ids = []
previous_text = ""

for new_tokens in llm.generate_stream(["Hello, world!"], sampling_params):
    for seq_id, token_ids in new_tokens:
        all_token_ids.append(token_ids)
        current_text = llm.tokenizer.decode(all_token_ids)
        delta = current_text[len(previous_text):]
        print(repr(delta))
        previous_text = current_text