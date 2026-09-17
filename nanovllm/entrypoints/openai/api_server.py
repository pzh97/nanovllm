import os
from fastapi import FastAPI
from nanovllm import LLM, SamplingParams
from .protocol import ChatMessage, ChatCompletionRequest
from .chat_template import apply_chat_template

app = FastAPI()
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(
    path, 
    enforce_eager=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = apply_chat_template(request)
    outputs =llm.generate(
        [prompt],
        sampling_params,
    )
    output_text = outputs[0]["text"]
    # return {
    #     "prompt": prompt
    # }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": "stop"
            }
        ]
    }

