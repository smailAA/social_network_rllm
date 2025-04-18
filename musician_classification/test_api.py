from openai import OpenAI


client = OpenAI(api_key="sk-m6iili3XRbC5jCd0360f9f96A4E94fC0B71257Fe8a568cF5", base_url="http://111.186.56.172:3000/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)