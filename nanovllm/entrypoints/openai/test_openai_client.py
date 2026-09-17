from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Hello from test client!"
        }
    ],
)

print(response.choices[0].message.content)

