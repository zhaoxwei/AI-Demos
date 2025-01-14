import os
import pickle
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the tokenizer and model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

def generate_embedding(text):
    """Generates an embedding for the given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def preprocess_query(query):
    """Preprocess the query to improve matching."""
    # Convert questions to keyword format
    question_words = {'what', 'who', 'where', 'when', 'how', 'why', 'do', 'does', 'am', 'is', 'are', 'have'}
    words = query.lower().split()
    keywords = [w for w in words if w not in question_words]
    return ' '.join(keywords)

def retrieve_relevant_documents(query, index_path='faiss_index', store_path='doc_store.pkl', similarity_threshold=0.01):
    """Retrieves relevant documents using both FAISS and TF-IDF."""
    try:
        index = faiss.read_index(index_path)
        with open(store_path, 'rb') as f:
            doc_store = pickle.load(f)
    except Exception as e:
        print(f"Error reading FAISS index or document store: {e}")
        return []

    # Preprocess query
    processed_query = preprocess_query(query)
    query_embedding = generate_embedding(processed_query)
    distances, indices = index.search(query_embedding, k=5)
    
    candidates = []
    candidate_indices = []
    for idx in indices[0]:
        if idx != -1 and idx in doc_store:
            candidates.append(doc_store[idx]['content'])
            candidate_indices.append(idx)
    
    if not candidates:
        return []

    # Improved vectorization
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),  # Use up to trigrams
        min_df=1,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True  # Apply sublinear TF scaling
    )
    
    # Vectorize documents
    tfidf_matrix = vectorizer.fit_transform([processed_query] + candidates)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    relevant_docs = []
    print("\nRelevant chunks:")
    for i, (similarity, idx) in enumerate(zip(cosine_similarities, candidate_indices)):
        chunk_info = doc_store[idx]
        
        print(f"\nFrom {chunk_info['source']} (Chunk {chunk_info['chunk_id']}):")
        print(f"Similarity: {similarity:.3f}")
        print(f"Content: {chunk_info['content']}")
        
        if similarity > 0 or i == 0:  # Include at least one result
            relevant_docs.append(chunk_info['content'])
    
    print(f"\nFound {len(relevant_docs)} relevant chunks")
    return relevant_docs
