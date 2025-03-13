import os
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # 新增导入

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

def retrieve_relevant_documents(query, index_path='faiss_index.bin', mapping_path='doc_store.pkl', k=5):
    """使用FAISS向量数据库检索相关文档"""
    try:
        # 加载FAISS索引
        index = faiss.read_index(index_path)
        
        # 加载内容映射（每次查询时重新加载）
        try:
            content_mapping = load_content_mapping(mapping_path)
        except Exception as e:
            print(f"Error loading mapping: {e}")
            content_mapping = {}

        # 生成查询向量并确保形状正确
        query_embedding = generate_embedding(query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # 验证维度是否匹配
        if query_embedding.shape[1] != index.d:
            raise ValueError(f"Embedding dimension {query_embedding.shape[1]} does not match index dimension {index.d}")
        
        # 执行搜索
        distances, indices = index.search(query_embedding, k)
        
        # 处理结果
        relevant_docs = []
        print("\nRelevant chunks:")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            try:
                idx = int(idx)
                content = content_mapping.get(idx, "Content not found")
            except KeyError:
                content = "Content not found"
            
            print(f"\nResult {i+1}:")
            print(f"Distance: {distance:.3f}")
            print(f"Index: {idx}")
            if content != "Content not found":
                relevant_docs.append(content.get('content'))  # 现在添加的是字符串
        
        print(f"\nFound {len(relevant_docs)} relevant chunks")
        return relevant_docs
        
    except Exception as e:
        print(f"Error searching FAISS: {e}")
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

# 在retrieve_relevant_documents函数之前添加
def load_content_mapping(mapping_path='doc_store.pkl'):
    """加载索引到内容的映射"""
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file {mapping_path} does not exist.")
    
    try:
        with open(mapping_path, 'rb') as f:
            return {int(k): v for k, v in pickle.load(f).items()}  # 确保key是int类型
    except Exception as e:
        raise RuntimeError(f"Failed to load mapping: {e}")


if __name__ == "__main__":
    # 测试代码
    test_query = "测试查询"
    results = retrieve_relevant_documents(test_query)