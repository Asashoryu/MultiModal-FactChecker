import chromadb
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB Persistent Storage
client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection("esg_vectors")

# Helper: UUID generator
def generate_uuid5(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

# Generate embedding
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Data ingestion functions
# def ingest_audio_data(audio_data):
#     """Store audio transcription data in ChromaDB."""
#     for audio in tqdm(audio_data, desc="Ingesting audio data"):
#         vector = get_embedding(audio['transcription'])
#         collection.add(
#             documents=[audio['transcription']],
#             metadatas=[audio],
#             ids=[generate_uuid5(audio['url'])],
#             embeddings=[vector]
#         )

def ingest_audio_data(audio_data):
    """Store ESG audio data in ChromaDB without duplicates."""
    for audio in tqdm(audio_data, desc="Ingesting audio data"):
        audio_id = generate_uuid5(audio['url'])

        # Check if this ID already exists
        existing_results = collection.query(
            query_texts=[audio['transcription']],
            n_results=1
        )

        if existing_results["ids"]:
            print(f"Skipping duplicate audio: {audio['url']}")
            continue  # Don't add duplicate

        vector = get_embedding(audio['transcription'])
        collection.add(
            documents=[audio['transcription']],
            metadatas=[audio],
            ids=[audio_id],
            embeddings=[vector]
        )

# def ingest_text_data(text_data):
#     """Store ESG report text in ChromaDB."""
#     for text in tqdm(text_data, desc="Ingesting text data"):
#         vector = get_embedding(text['text'])
#         collection.add(
#             documents=[text['text']],
#             metadatas=[text],
#             ids=[generate_uuid5(f"{text['source_document']}_{text['page_number']}_{text['paragraph_number']}")],
#             embeddings=[vector]
#         )

def ingest_text_data(text_data):
    """Store ESG report text in ChromaDB without duplicates."""
    for text in tqdm(text_data, desc="Ingesting text data"):
        text_id = generate_uuid5(f"{text['source_document']}_{text['page_number']}_{text['paragraph_number']}")

        # Check if this ID already exists
        existing_results = collection.query(
            query_texts=[text["text"]],
            n_results=1
        )

        if existing_results["ids"]:
            print(f"Skipping duplicate text: {text['source_document']} Page {text['page_number']}")
            continue  # Don't add duplicate

        vector = get_embedding(text["text"])
        collection.add(
            documents=[text["text"]],
            metadatas=[text],
            ids=[text_id],
            embeddings=[vector]
        )


# def ingest_image_data(image_data):
#     """Store ESG images in ChromaDB without requiring a description."""
#     for image in tqdm(image_data, desc="Ingesting image data"):
#         vector = get_embedding(image["image_path"])  # Use `image_path` instead

#         collection.add(
#             documents=[image["image_path"]],  # Store image path instead of description
#             metadatas=[image],  # Retain original metadata
#             ids=[generate_uuid5(f"{image['source_document']}_{image['page_number']}_{image['image_path']}")],
#             embeddings=[vector]
#         )

def ingest_image_data(image_data):
    """Store ESG images in ChromaDB while preventing duplicates."""
    for image in tqdm(image_data, desc="Ingesting image data"):
        image_id = generate_uuid5(f"{image['source_document']}_{image['page_number']}_{image['image_path']}")

        # Check if this ID already exists
        existing_results = collection.query(
            query_texts=[image["image_path"]],
            n_results=1
        )

        if existing_results["ids"]:
            print(f"Skipping duplicate image: {image['image_path']}")
            continue  # Don't add duplicate

        vector = get_embedding(image["image_path"])
        collection.add(
            documents=[image["image_path"]],
            metadatas=[image],
            ids=[image_id],
            embeddings=[vector]
        )


# def ingest_table_data(table_data):
#     """Store ESG tables in ChromaDB using `table_content` instead of `description`."""
#     for table in tqdm(table_data, desc="Ingesting table data"):
#         vector = get_embedding(table["table_content"])  # Use `table_content` instead

#         collection.add(
#             documents=[table["table_content"]],  # Store full table content
#             metadatas=[table],  # Retain full metadata
#             ids=[generate_uuid5(f"{table['source_document']}_{table['page_number']}")],
#             embeddings=[vector]
#         )

def ingest_table_data(table_data):
    """Store ESG tables in ChromaDB while preventing duplicates."""
    for table in tqdm(table_data, desc="Ingesting table data"):
        table_id = generate_uuid5(f"{table['source_document']}_{table['page_number']}")

        # Check if this ID already exists
        existing_results = collection.query(
            query_texts=[table["table_content"]],
            n_results=1
        )

        if existing_results["ids"]:
            print(f"Skipping duplicate table from Page {table['page_number']}")
            continue  # Don't add duplicate

        vector = get_embedding(table["table_content"])
        collection.add(
            documents=[table["table_content"]],
            metadatas=[table],
            ids=[table_id],
            embeddings=[vector]
        )



# Unified ingestion function
def ingest_all_data(audio_data, text_data, image_data, table_data):
    """Store all multimodal ESG data in ChromaDB."""
    ingest_audio_data(audio_data)
    ingest_text_data(text_data)
    ingest_image_data(image_data)
    ingest_table_data(table_data)

# Multimodal search function
def search_multimodal(query: str, limit: int = 10):
    """Perform vector search in ChromaDB to retrieve relevant ESG data."""
    query_vector = get_embedding(query)
    results = collection.query(query_embeddings=[query_vector], n_results=limit)

    return results
