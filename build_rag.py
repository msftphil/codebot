# build_rag.py

import argparse
import os
import logging
import sys
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import filetype
import pickle

def load_file_with_filetype(file_path):
    """Load a file and return a Document object with content and metadata."""
    # Detect file type using filetype
    kind = filetype.guess(file_path)

    # Fallback to checking file extension if filetype fails
    if not kind:
        print(f"Could not detect file type for: {file_path}")
        # Assume common text file extensions are valid
        if file_path.endswith(('.txt', '.py', '.md', '.gitignore', '.json', '.csv', '.xml')):
            print(f"Treating as text file based on extension: {file_path}")
        else:
            print(f"Skipping file with unknown type: {file_path}")
            return None

    # Only process text-based files
    if kind and not "text" in kind.mime and not "json" in kind.mime:
        print(f"Skipping non-text file: {file_path} ({kind.mime})")
        return None

    # Attempt to read the file as text
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return Document(page_content=content, metadata={"source": file_path})
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


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
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            logger.info(f"Processing file: {file_path}")
            try:
                # Load the document
                doc = load_file_with_filetype(file_path)
                if doc:
                    # Add metadata to the document
                    doc.metadata["filename"] = file
                    doc.metadata["filepath"] = file_path
                    doc.metadata["directory"] = root
                    documents.append(doc)
                    processed_files += 1
                else:
                    skipped_files += 1
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
