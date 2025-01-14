import os
from document_ingestion import ingest_documents

def main():
    """CLI for ingesting documents into the knowledge base."""
    # Default to 'pdf' directory in current working directory
    default_dir = os.path.join(os.getcwd(), 'pdf')
    directory = input(f"Enter the path to the directory containing documents (default: {default_dir}): ").strip()
    
    if not directory:
        directory = default_dir
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        create = input("Create directory? (y/n): ").lower()
        if create == 'y':
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            return
    
    try:
        documents = ingest_documents(directory)
        print(f"Successfully processed documents from {directory}")
    except Exception as e:
        print(f"Error processing documents: {e}")

if __name__ == "__main__":
    main()
