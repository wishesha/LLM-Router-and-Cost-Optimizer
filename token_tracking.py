from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-10-21"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant"}
]

print("Chat Started (Type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model = deployment,
        messages = messages
    )

    reply = response.choices[0].message.content
    print(f"Assistant {reply}")

    messages.append({"role": "assistant", "content": reply})

    print(f"Current message count in history: {len(messages)}")