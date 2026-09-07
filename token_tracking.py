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

MAX_RESPONSE_TOKENS = 500
MAX_WINDOW = 4
running_summary = ""
summary_system_instructions = {"role": "system", "content": ("You are a memory compressor for a chat system. Your task is to merge new conversation turns in an existing running list. "
"A few rules: 1. Synthesize concepts into high-level topics instead of listing questions and answers. 2. Never simply append every single turn verbatim. "
"3. Prioritize user facts and goals over assistant replies. 4. Keep responses brief but informative(under 40 words)")}
system_instructions = {"role": "system", "content": f"You are a helpful assistant. Context from past turns: {running_summary}"}
chat_history = []


print("Chat Started (Type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    chat_history.append({"role": "user", "content": user_input})
    recent_history = chat_history[-MAX_WINDOW:]
    

    model_context = [system_instructions] + recent_history
    print(f"[DEBUG]\nMessage count in history: {len(chat_history)}\nMessage count sent to AI: {len(model_context)}\n")

    response = client.chat.completions.create(
        model = deployment,
        messages = model_context,
        max_completion_tokens=MAX_RESPONSE_TOKENS
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
                {"role": "user", "content": f"Running Summary: {running_summary}. New turn to summarize with it: {text_to_summarize}"}
            ],
            max_completion_tokens=100
        )

        new_summary = summary_response.choices[0].message.content

        if new_summary and new_summary.strip():
            running_summary = new_summary
            
        print(f"\n[DEBUG]\nRunning summary: {running_summary}")

    system_instructions = {
        "role": "system", 
        "content": f"You are a helpful assistant. Context from past turns: {running_summary}"
    }