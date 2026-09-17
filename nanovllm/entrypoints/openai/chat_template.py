from transformers import AutoTokenizer

from .protocol import ChatMessage, ChatCompletionRequest

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def apply_chat_template(request: ChatCompletionRequest) -> str:
    messages = [
        message.model_dump() for message in request.messages
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

if __name__ == "__main__":
    request = ChatCompletionRequest(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                ChatMessage(
                    role="user",
                    content="hi",
                )
            ],
            stream=True,
        )

    prompt = apply_chat_template(request)
    print(prompt)
