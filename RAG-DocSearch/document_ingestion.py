import os
import sys
import pickle
from PyPDF2 import PdfReader
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Initialize the tokenizer and model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

def read_pdf(file_path):
    """Reads the content of a PDF document."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            content = ""
            for page in reader.pages:
                content += page.extract_text()
            return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def read_document(file_path):
    """Reads the content of a document."""
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def process_document(content):
    """Processes the content of a document."""
    # Placeholder for document processing logic
    return content

def generate_embedding(text):
    """Generates an embedding for the given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def save_embeddings_to_faiss(embeddings, index_path='faiss_index'):
    """Saves the embeddings to a FAISS index."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print("Embeddings saved to FAISS index.")

def split_into_chunks(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def ingest_documents(directory):
    """Ingests all documents in a directory."""
    chunks = []
    embeddings = []
    doc_store = {}
    chunk_id = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            content = read_document(file_path)
            if content is not None:
                # Split content into chunks
                content_chunks = split_into_chunks(content)
                for chunk in content_chunks:
                    processed_chunk = process_document(chunk)
                    embedding = generate_embedding(processed_chunk)
                    embeddings.append(embedding)
                    doc_store[chunk_id] = {
                        'content': processed_chunk,
                        'source': filename,
                        'chunk_id': chunk_id
                    }
                    chunks.append(processed_chunk)
                    chunk_id += 1
                print(f"Processed {filename}: {len(content_chunks)} chunks")
    
    if embeddings:
        embeddings = np.vstack(embeddings)
        save_embeddings_to_faiss(embeddings)
        with open('doc_store.pkl', 'wb') as f:
            pickle.dump(doc_store, f)
        print(f"Saved {len(doc_store)} chunks to store")
    
    return chunks

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python document_ingestion.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    documents = ingest_documents(directory)
    for filename, content in documents.items():
        print(f"Document: {filename}\nContent: {content}\n")
