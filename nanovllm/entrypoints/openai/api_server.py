import os
import uuid
import time
import json
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
from nanovllm import LLM, SamplingParams
from .protocol import ChatCompletionRequest
from .chat_template import apply_chat_template, decode_output, count_tokens

app = FastAPI()
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(
    path, 
    enforce_eager=True,
    tensor_parallel_size=1,
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    prompt = apply_chat_template(request)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    if request.stream:
        def generate_stream():
            all_tokens_ids = []
            previous_text = ""
            for new_tokens, finish_reason in llm.generate_stream([prompt], sampling_params):
                for seq_id, token_id in new_tokens:
                    all_tokens_ids.append(token_id)
                    current_text = llm.tokenizer.decode(all_tokens_ids, skip_special_tokens=True)
                    text = current_text[len(previous_text):]
                    previous_text = current_text
                    if not text:
                        continue
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": text
                                },
                                "finish_reason": None
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            done_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }
                 ],
            }

            yield f"data: {json.dumps(done_chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate_stream(), media_type="text/event-stream",)
    outputs =llm.generate(
        [prompt],
        sampling_params,
    )
    token_ids = outputs[0]["token_ids"]
    output_text = decode_output(token_ids)
    prompt_tokens = count_tokens(prompt)
    completion_tokens = len(token_ids)
    total_tokens = prompt_tokens + completion_tokens
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text
                },
                "finish_reason": outputs[0]['finish_reason']
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    }

