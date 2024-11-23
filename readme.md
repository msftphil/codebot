# RAG Chatbot with LangChain

This repository contains scripts to build and run a Retrieval-Augmented Generation (RAG) chatbot using LangChain. The project focuses on leveraging language models to provide context-aware and relevant responses, making it suitable for applications requiring information retrieval and conversational AI.

## Table of Contents
* [File Descriptions](#FileDescriptions)
	* [build_rag.py](#build_rag.py)
	* [chatbot.py](#chatbot.py)
	* [ollama_llm.py](#ollama_llm.py)
* [Dependencies](#Dependencies)
* [Usage](#Usage)
	* [Running the Scripts](#RunningtheScripts)
	* [Build the RAG Pipeline](#BuildtheRAGPipeline)
	* [Run the Chatbot](#RuntheChatbot)
* [Notes](#Notes)

## File Descriptions

### build_rag.py
This script is responsible for constructing the RAG pipeline. It defines how the language model interacts with a retrieval mechanism to provide accurate and contextual responses. The script outlines the embedding generation, retrieval, and model configuration.

### chatbot.py
This script serves as the entry point for interacting with the RAG chatbot. It defines the user interface, input/output handling, and logic for querying the RAG pipeline constructed in build_rag.py.

### ollama_llm.py
This files provides and interface between OLLAMA and the langchain backend.

## Dependencies
The project requires Python 3.8 or higher, and Ollama (https://ollama.com/)

Recommend using llama3.2 (https://ollama.com/library/llama3.2) and/or codellama (https://ollama.com/library/codellama) models

For additional details w.r.t. LangChain, see (https://www.langchain.com/) for additional details, but the TLDR is that LangChain is a framework for developing applications powered by language models. It provides abstractions for chaining together multiple LLM calls and integrating them with external tools.

  
## Usage

Python dependencies can be installed using the following command:

    pip install -r requirements.txt

### Build the RAG Pipeline
Run build_rag.py to set up the Retrieval-Augmented Generation pipeline. This step might include creating embeddings, configuring retrieval mechanisms, and ensuring the language model is properly integrated:

    python3 build_rag.py --ouptut codebot.pkl ./

### Run the Chatbot
Launch the chatbot interface by running chatbot.py:

    python3 chatbot.py --vectorstore codebot.pkl

## Notes
Scripts assume that you've already installed ollama and acquired a model (such as those recommended above), and that the ollama server is running (check via ollama ps to confirm).