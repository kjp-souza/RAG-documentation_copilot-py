import os
import openai
import requests
import json
import yaml  # Added YAML support
import faiss
import chromadb
import yaml
import datetime
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Change to this to use OpenAI embeddings
# from langchain_openai import OpenAIEmbeddings

# Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ollama API URL
OLLAMA_URL = "http://localhost:11434/api/generate"

# Initialize FAISS index for vector search
index = None  

def load_yaml(file_path):
    """Load a YAML file and handle datetime serialization."""
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    # Convert datetime objects to string before converting to JSON
    def datetime_converter(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        raise TypeError("Type not serializable")

    # Create a Document with content (string) from the YAML data
    return Document(page_content=json.dumps(yaml_data, indent=2, default=datetime_converter), metadata={'source': file_path})

def load_documents_from_directory(directory):
    """Load documents from a directory (PDF, Markdown, and YAML)."""
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs.extend(loader.load())
            elif file.endswith(".md") or file.endswith(".txt"):
                loader = TextLoader(file_path)
                docs.extend(loader.load())
            elif file.endswith(".yaml") or file.endswith(".yml"):
                # Load YAML file and return as Document
                docs.append(load_yaml(file_path))
            else:
                continue  # Skip unsupported files
    
    return docs

def split_and_embed(docs):
    """Splits text into chunks and embeds them into FAISS."""
    global index
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # embeddings = OpenAIEmbeddings()  # Change to this to use the OpenAI embedding model
    # Here, we are using an open-source embedding model:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en", model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_documents(chunks, embeddings)

    index = vector_store  # Store the FAISS index

def retrieve_context(query, k=3):
    """Retrieves top-k most relevant document chunks."""
    if index is None:
        print("No knowledge base loaded. Please upload documents first.")
        return []

    results = index.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

def ask_openai(prompt, context=""):
    """Send a prompt to OpenAI with additional context."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt}"},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI request error: {e}")
        return None

def ask_ollama(prompt, model="mistral"):
    """Send a prompt to the Ollama API (supports DeepSeek-R1 7B)."""
    payload = {"model": model, "prompt": prompt}

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()  

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    full_response += data.get("response", "")
                    if data.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    print(f"JSON decoding error: {e}")
        return full_response

    except requests.RequestException as e:
        print(f"Ollama request error: {e}")
        return None

def main():
    global index
    print("Welcome to the RAG Documentation Copilot!")

    # Load knowledge base at startup
    input_path = input("Enter a file path or directory for knowledge base: ").strip()

    if os.path.isdir(input_path):
        print(f"Loading all documents from directory: {input_path}")
        docs = load_documents_from_directory(input_path)
    elif os.path.isfile(input_path):
        docs = load_documents_from_directory(os.path.dirname(input_path))  # Load just one file
    else:
        print("Invalid path. Please provide a valid file or directory.")
        return

    split_and_embed(docs)
    print("Knowledge base loaded successfully.")

    while True:
        user_input = input("\nYour question: ")
        api_choice = input("Choose API (openai/ollama/deepseek-r1): ").strip().lower()

        # Retrieve relevant knowledge
        context = retrieve_context(user_input)

        if api_choice == "openai":
            response = ask_openai(user_input, context)
        elif api_choice == "deepseek-r1":
            response = ask_ollama(user_input, model="deepseek-r1")
        else:
            response = ask_ollama(user_input, model="mistral")  # Default Ollama model

        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()
