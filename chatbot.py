# chatbot.py

import argparse
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from ollama_llm import Ollama  # Custom LLM class for Ollama

def main():
    parser = argparse.ArgumentParser(description='Chatbot that uses RAG database to ground responses.')
    parser.add_argument('--vectorstore', type=str, default='faiss_store.pkl', help='Vectorstore file.')
    parser.add_argument('--model', type=str, default='llama3.2', help='Ollama model name.')
    parser.add_argument('--base_url', type=str, default='http://localhost:11434', help='Base URL for Ollama API.')
    args = parser.parse_args()

    # Load vectorstore from disk
    with open(args.vectorstore, 'rb') as f:
        vectorstore = pickle.load(f)

    # Set up retriever
    retriever = vectorstore.as_retriever()

    # Set up LLM (Ollama)
    llm = Ollama(model_name=args.model, base_url=args.base_url)

    # Set up memory with output_key specified
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer'  # Specify which output to store in memory
    )

    # Define a custom prompt template to include context
    from langchain.prompts import PromptTemplate

    prompt_template = """
    Use the following context to answer the question.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Set up Conversational Retrieval Chain with custom prompt
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    print("Type 'exit' to quit the chatbot.")

    while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break

        # Run the query through the chain using invoke()
        result = qa.invoke({"question": query})

        answer = result['answer']
        print(f"Assistant: {answer}")

        # Optional: Print the source documents and their metadata
        # print("\nSource Documents:")
        # for doc in result['source_documents']:
        #     print(f"Filename: {doc.metadata.get('filename')}")
        #     print(f"Filepath: {doc.metadata.get('filepath')}\n")

if __name__ == "__main__":
    main()
