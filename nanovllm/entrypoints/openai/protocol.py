from typing import Literal
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False 

if __name__ == "__main__":
    request = ChatCompletionRequest(
        model="gpt-4",
        messages=[
            ChatMessage(
                role="user",
                content="hi",
            )
        ],
        stream=True,
    )
    messages = [message.model_dump() for message in request.messages]
    print(messages)

    print(request)
    print(request.model)
    print(request.messages[0].model_dump())
    print(request.stream)
