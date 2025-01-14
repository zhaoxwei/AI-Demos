from retrieval import retrieve_relevant_documents
from generation import generate_response

def query_knowledge_base():
    """Query the knowledge base and generate a response."""
    query = input("Enter your query: ")
    relevant_docs = retrieve_relevant_documents(query)
    
    if relevant_docs:
        context = " ".join(relevant_docs)
        prompt = f"{context}\n\n{query}"
        print("Context found in knowledge base. Generating response...")
        response = generate_response(prompt)
        print("\nResponse:", response)
    else:
        print("No relevant information found in knowledge base.")

def main():
    while True:
        query_knowledge_base()
        if input("\nAsk another question? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()
