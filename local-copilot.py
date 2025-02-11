import openai
import os
import requests
import json
from openai import OpenAIError  # Import OpenAIError directly

# Set the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Base URL for the local Ollama API
OLLAMA_URL = 'http://localhost:11434/api/generate'

def ask_openai(prompt):
    """Send a prompt to the OpenAI API and return the response."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        # Access the 'content' attribute directly
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"OpenAI request error: {e}")
        return None

def ask_ollama(prompt, model="deepseek-r1"):
    """Send a prompt to the Ollama API and return the response."""
    payload = {"model": model, "prompt": prompt}

    try:
        # Stream the response to handle large responses
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()  # Ensure the request was successful

        # Collect the response text from Ollama
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    # Parse each line as JSON
                    data = json.loads(line)
                    # Append the response part
                    full_response += data.get("response", "")
                    # Stop if "done" is True in the response
                    if data.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}")
        return full_response

    except requests.RequestException as e:
        print(f"Ollama request error: {e}")
        return None

def ask(prompt, use_openai=True):
    """Choose which API to use based on the `use_openai` flag."""
    if use_openai:
        return ask_openai(prompt)
    else:
        return ask_ollama(prompt)

# Main interactive loop
while True:
    user_input = input("Your question: ")
    api_choice = input("Choose API (openai/ollama): ").strip().lower()
    use_openai = api_choice == "openai"  # Flag to decide between OpenAI and Ollama

    response = ask(user_input, use_openai=use_openai)
    print(f"Assistant: {response}")
