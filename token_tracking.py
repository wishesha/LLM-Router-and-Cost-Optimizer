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

MAX_RESPONSE_TOKENS = 800
MAX_SUMMARY_TOKENS = 400
MAX_WINDOW = 4
running_summary = ""
summary_system_instructions = {"role": "system", "content": ("You are a memory compressor for a chat system. Focus only on the following: 1. Specific topics discussed and what seems to be important to the user."
"2. Active constraints given by user (response length or kind of response). 3. DO NOT simply restate the questions and their answers. They must be synthesized with the previous running summary you are receiving to make a cohesive explaination yet still succinct. "
"4. Keep responses under 60 words maximum.")}

system_instructions = {"role": "system", "content": f"You are a helpful assistant. Context from past turns: {running_summary}. Keep responses under 2-3 sentences unless otherwise stated."}
chat_history = []


print("Chat Started (Type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    chat_history.append({"role": "user", "content": user_input})
    recent_history = chat_history[-MAX_WINDOW:]
    
    system_instructions = {
            "role": "system", 
            "content": f"You are a helpful assistant. Context from past turns: {running_summary}"
        }
    
    model_context = [system_instructions] + recent_history
    print("Message count in history: {len(chat_history)}\nMessage count sent to AI: {len(model_context)}\n")

    response = client.chat.completions.create(
        model = deployment,
        messages = model_context,
        max_completion_tokens=MAX_RESPONSE_TOKENS,
        reasoning_effort = "low"
    )

    reply = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    print(f"Assistant: {reply}")
    if finish_reason == "length":
        print(f"\nResponse could not be completed because it reached max completion tokens")

    chat_history.append({"role": "assistant", "content": reply})

    dropped_messages = chat_history[:-MAX_WINDOW]
    
    text_to_summarize = ""
    if dropped_messages:
        for message in dropped_messages[-2:]:
            text_to_summarize += f"{message['role']}: {message['content']}\n"

        summary_response = client.chat.completions.create(
            model = deployment,
            messages = [
                summary_system_instructions,
                {"role": "user", "content": f"Running Summary: {running_summary}. New turn to summarize with it: {text_to_summarize}\n"}
            ],
            max_completion_tokens= MAX_SUMMARY_TOKENS,
            reasoning_effort = "low"
        )

        new_summary = summary_response.choices[0].message.content

        if new_summary and new_summary.strip():
            running_summary = new_summary
            
        print(f"\n[DEBUG]\nRunning summary: {running_summary}")

    