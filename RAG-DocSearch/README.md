# RAG Application

A Retrieval-Augmented Generation (RAG) application for document Q&A.

## Components & Dependencies

### Text Embedding
- **Model**: DistilBERT (distilbert-base-uncased)
- **Library**: transformers==4.35.2
- **Framework**: PyTorch
- **Purpose**: Document and query embedding generation
- **Implementation**: Mean pooling of last hidden states
- **Output Dimension**: 768

### Language Model
- **Model**: Qwen 2.5 (1.5B parameters)
- **Library**: ollama-python==0.1.6
- **Deployment**: Local Ollama server
- **API Endpoint**: http://127.0.0.1:11434
- **Input Format**: JSON with messages array
- **Context Window**: 8192 tokens

### Vector Search (FAISS)
- **Library**: faiss-cpu==1.7.4
- **Index Type**: IndexFlatL2
- **Distance Metric**: L2 (Euclidean)
- **Dimension**: 768 (matches DistilBERT)
- **Top-k**: 5 documents
- **Implementation**: `faiss.IndexFlatL2`

### Text Processing
- **PDF Processing**: PyPDF2==3.0.1
- **Tokenization**: 
  - Primary: transformers.AutoTokenizer
  - Secondary: scikit-learn TfidfVectorizer
- **Text Chunking**:
  - Chunk size: 1000 characters
  - Overlap: 100 characters
  - Implementation: Custom sliding window

### TF-IDF Processing
- **Library**: scikit-learn==1.3.2
- **Vectorizer**: TfidfVectorizer
- **Configuration**:
  - n-grams: (1,3)
  - min_df: 1
  - max_df: 0.9
  - strip_accents: unicode
  - sublinear_tf: True

### Storage
- **Vector Store**: FAISS index file
  - Format: Binary index
  - Filename: 'faiss_index'
- **Document Store**: Pickle file
  - Format: Python dictionary
  - Filename: 'doc_store.pkl'
  - Metadata: source, chunk_id, content

## Project Structure

```
ragapp/
├── app.py              # Query interface
├── ingest.py           # Document ingestion CLI
├── document_ingestion.py  # Document processing & embedding
├── retrieval.py        # Search combining FAISS & TF-IDF
├── generation.py       # Ollama LLM integration
└── README.md
```

## Dependencies Installation

```bash
pip install transformers==4.35.2 torch faiss-cpu==1.7.4 PyPDF2==3.0.1 
pip install scikit-learn==1.3.2 ollama-python==0.1.6
```

## Usage

1. Start Ollama server:
```bash
ollama serve
```

2. Pull Qwen model:
```bash
ollama pull qwen2.5:1.5b
```

3. Ingest documents:
```bash
python ingest.py
```

4. Query the knowledge base:
```bash
python app.py
```

## Technical Implementation Details

### Document Processing Pipeline
1. PDF Extraction (PyPDF2)
2. Text Chunking (sliding window)
3. Embedding Generation (DistilBERT)
4. Vector Storage (FAISS)
5. Metadata Storage (Pickle)

### Retrieval Process
1. Query Processing
   - Question word filtering
   - Embedding generation
2. Vector Search (FAISS)
   - L2 distance calculation
   - Top-k retrieval
3. TF-IDF Ranking
   - N-gram matching
   - Cosine similarity
4. Score Combination
   - Weighted average
   - Threshold filtering

### Response Generation
1. Context Assembly
   - Relevant chunk collection
   - Context formatting
2. LLM Integration
   - Ollama API call
   - Response parsing
3. Error Handling
   - Connection retry
   - Fallback responses

## Performance Considerations
- Embedding Dimension: 768
- Average Chunk Size: 1000 chars
- Vector Search Complexity: O(n)
- Memory Usage: ~500MB per 1M vectors
- Response Time: 
  - Retrieval: ~100ms
  - Generation: ~1-2s