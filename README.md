# Multiple LLM copilot with Retrieval Augmented Generation (RAG)

The first script `local-copilot.py` can call multiple LLM APIs (currently using only [OpenAI](https://platform.openai.com/docs/overview) and [Ollama](https://ollama.com/)) locally.

The second script `rag-copilot.py` allows ingesting relevant content as knowledge base. 
You can add your relevant documents, such as PDF, YAML or Markdown files.

## Call LLMs locally

### Getting Started:

-   Create a [Python](https://docs.python.org/3/library/venv.html) environment or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment using an up-to-date Python version. This project uses `Python version 3.11.5`. You can use a command, such as: `conda create -n myenv python=3.11.5`.

-   Install the `requirements.txt`

-   Edit the .env file with your OpenAI or other API keys and export it with a command, such as `export OPENAI_API_KEY="replace-with-your-openai-key"`

-   Call the LLMs by running the script `python3 local-copilot.py` choosing to ask either Ollama or OpenAI.
    -   While using Ollama, you can pull any LLM model by running `ollama pull 'model-name'`, e.g. `ollama pull deepseek-r1`.

## Call LLMs using a RAG knowledge base

RAG allows ingesting relevant content based on your needs so the LLMs can provide more relavant and up-to-date answers based on the knowledge you wish to ingest. This script accepts PDF, Markdown or YAML files. It uses [FAISS vector indexing - Langchain's integration](https://python.langchain.com/docs/integrations/vectorstores/faiss/) and open-source [HuggingFace sentence transformers](https://huggingface.co/BAAI/bge-m3) or [OpenAI](https://platform.openai.com/docs/overview) embedding models.

### Getting Started:

-   Create a [Python](https://docs.python.org/3/library/venv.html) environment or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment using an up-to-date Python version. This project uses `Python version 3.11.5`. You can use a command, such as: `conda create -n myenv python=3.11.5`.

-   Install the `requirements2.txt`

-   Call the LLMs by running the script `python3 rag-copilot.py`

-   Ingest new documents by passing the folder name, e.g.: `DeviceLocation-main`

-   Choose to prompt either Ollama or OpenAI.
    -   While using Ollama, you can pull any LLM model by running `ollama pull 'model-name'`, e.g. `ollama pull deepseek-r1`.

## License

This project is licensed under the Apache 2.0 - see the `LICENSE` file for details.
