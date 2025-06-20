import weaviate
from weaviate.classes import Property, DataType
import weaviate.classes.query as wq
from tqdm import tqdm
import uuid
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Helper: UUID generator
def generate_uuid5(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))

# Connect to Weaviate Cloud Service
WCS_URL = os.getenv("WCS_URL")  # Ensure these are set in your environment
WCS_API_KEY = os.getenv("WCS_API_KEY")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WCS_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCS_API_KEY),
)

# Define metadata schema
properties = [
    Property(name="source_document", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="page_number", data_type=DataType.INT, skip_vectorization=True),
    Property(name="paragraph_number", data_type=DataType.INT, skip_vectorization=True),
    Property(name="text", data_type=DataType.TEXT),
    Property(name="image_path", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="description", data_type=DataType.TEXT),
    Property(name="base64_encoding", data_type=DataType.BLOB, skip_vectorization=True),
    Property(name="table_content", data_type=DataType.TEXT),
    Property(name="url", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="audio_path", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="transcription", data_type=DataType.TEXT),
    Property(name="content_type", data_type=DataType.TEXT, skip_vectorization=True),
]

# Create collection if not exists
def initialize_collection():
    if "RAGESGDocuments" in client.collections.list_all():
        client.collections.delete("RAGESGDocuments")

    client.collections.create(
        name="RAGESGDocuments",
        properties=properties,
        vectorizer_config=None
    )

# Generate embedding
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Data ingestion functions
def ingest_audio_data(collection, audio_data):
    with collection.batch.dynamic() as batch:
        for audio in tqdm(audio_data, desc="Ingesting audio data"):
            vector = get_embedding(audio['transcription'])
            batch.add_object(
                properties={**audio, "content_type": "audio"},
                uuid=generate_uuid5(audio['url']),
                vector=vector
            )

def ingest_text_data(collection, text_data):
    with collection.batch.dynamic() as batch:
        for text in tqdm(text_data, desc="Ingesting text data"):
            vector = get_embedding(text['text'])
            batch.add_object(
                properties={**text, "content_type": "text"},
                uuid=generate_uuid5(f"{text['source_document']}_{text['page_number']}_{text['paragraph_number']}"),
                vector=vector
            )

def ingest_image_data(collection, image_data):
    with collection.batch.dynamic() as batch:
        for image in tqdm(image_data, desc="Ingesting image data"):
            vector = get_embedding(image['description'])
            batch.add_object(
                properties={**image, "content_type": "image"},
                uuid=generate_uuid5(f"{image['source_document']}_{image['page_number']}_{image['image_path']}"),
                vector=vector
            )

def ingest_table_data(collection, table_data):
    with collection.batch.dynamic() as batch:
        for table in tqdm(table_data, desc="Ingesting table data"):
            vector = get_embedding(table['description'])
            batch.add_object(
                properties={**table, "content_type": "table"},
                uuid=generate_uuid5(f"{table['source_document']}_{table['page_number']}"),
                vector=vector
            )

# Unified ingestion function
def ingest_all_data(collection_name, audio_data, text_data, image_data, table_data):
    collection = client.collections.get(collection_name)
    ingest_audio_data(collection, audio_data)
    ingest_text_data(collection, text_data)
    ingest_image_data(collection, image_data)
    ingest_table_data(collection, table_data)

# Multimodal search function
def search_multimodal(query: str, limit: int = 10):
    query_vector = get_embedding(query)
    collection = client.collections.get("RAGESGDocuments")
    return collection.query.near_vector(
        near_vector=query_vector,
        limit=limit,
        return_metadata=wq.MetadataQuery(distance=True),
        return_properties=[
            "content_type", "url", "audio_path", "transcription",
            "source_document", "page_number", "paragraph_number", "text",
            "image_path", "description", "table_content"
        ]
    ).objects

# Function to delete collection before running ingestion
def reset_collection():
    """Deletes the Weaviate collection RAGESGDocuments if it exists."""
    if "RAGESGDocuments" in client.collections.list_all():
        client.collections.delete("RAGESGDocuments")
        print("RAGESGDocuments collection has been deleted.")
    else:
        print("Collection RAGESGDocuments does not exist, skipping deletion.")
