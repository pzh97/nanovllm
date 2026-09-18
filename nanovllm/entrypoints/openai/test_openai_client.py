from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[
        {
            "role": "user",
            "content": "explain what a transformer is in three sentences."
        }
    ],
    max_tokens=3,
    stream=False,
)

print("content =", repr(response.choices[0].message.content))
print("finish =", response.choices[0].finish_reason)

# for chunk in stream:
#     print(
#         "content =", repr(chunk.choices[0].delta.content),
#         "finish =", chunk.choices[0].finish_reason,
#     )
