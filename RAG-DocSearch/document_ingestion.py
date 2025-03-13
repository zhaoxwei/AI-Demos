import os
import sys
import pickle
import numpy as np
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModel
import json
import faiss  # 新增FAISS依赖

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
    """Saves the embeddings and documents to Redis."""
    client = Redis(host=redis_host, port=redis_port, db=redis_db)
    
    # Create Redis index for vector similarity search
    try:
        schema = (
            VectorField("embedding", "FLOAT32", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": embeddings.shape[1],
                "DISTANCE_METRIC": "L2"
            }),
            TextField("content"),
            TextField("source"),
            TextField("chunk_id")
        )
        client.ft("doc_idx").create_index(schema)
    except Exception as e:
        print(f"Index might already exist: {e}")
    
    # Store documents and embeddings
    pipe = client.pipeline()
    for idx, (embedding, doc_info) in enumerate(zip(embeddings, doc_store.values())):
        key = f"doc:{idx}"
        vector_data = {
            'embedding': embedding.tobytes(),
            'content': doc_info['content'],
            'source': doc_info['source'],
            'chunk_id': str(doc_info['chunk_id'])
        }
        pipe.hset(key, mapping=vector_data)
    pipe.execute()
    print(f"Saved {len(doc_store)} documents to Redis")

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

# 需要先安装faiss：pip install faiss-cpu

def save_to_faiss(embeddings, doc_store):
    """将embeddings保存到FAISS索引"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # 保存索引和文档存储
    faiss.write_index(index, "faiss_index.bin")
    with open('doc_store.pkl', 'wb') as f:
        pickle.dump(doc_store, f)
    print(f"Saved {len(doc_store)} documents to FAISS")

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
        save_to_faiss(embeddings, doc_store)  # 替换原来的save_to_redis调用
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
