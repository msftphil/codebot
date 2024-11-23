# build_rag.py

import argparse
import os
import logging
import sys
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

def main():
    parser = argparse.ArgumentParser(description='Build a RAG database from documents in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing documents to process.')
    parser.add_argument('--output', type=str, default='faiss_store.pkl', help='Output file for the vectorstore.')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(__name__)

    documents = []
    total_files = 0
    processed_files = 0
    skipped_files = 0

    # Walk through the directory and process files
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            logger.info(f"Processing file: {file_path}")
            try:
                # Load the document
                loader = UnstructuredFileLoader(file_path)
                doc = loader.load()
                
                # Add metadata to the document
                for d in doc:
                    d.metadata['filename'] = file
                    d.metadata['filepath'] = file_path
                    d.metadata['directory'] = root
                
                documents.extend(doc)
                processed_files += 1
            except Exception as e:
                logger.warning(f"Error loading file {file_path}: {e}")
                skipped_files += 1
                continue

    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully processed files: {processed_files}")
    logger.info(f"Skipped files due to errors: {skipped_files}")

    if not documents:
        logger.error("No documents were loaded. Exiting.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(docs)} chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore to disk
    with open(args.output, 'wb') as f:
        pickle.dump(vectorstore, f)
    logger.info(f"Vectorstore saved to {args.output}")

if __name__ == "__main__":
    main()
