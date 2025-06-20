import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_storage")

def reset_chroma():
    """Deletes all data from ChromaDB and resets storage."""
    client.delete_collection("esg_vectors")
    client.get_or_create_collection("esg_vectors")
    print("ChromaDB storage has been reset.")
    
if __name__ == "__main__":
    reset_chroma()
