from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Who are you?"
        }
    ]
)

print(completion.choices[0].message)