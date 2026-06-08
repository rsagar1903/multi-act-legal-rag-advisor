import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import os

def embed_chunks():
    # Load chunks
    with open("data/bns_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(
        path="chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Clear existing collection
    try:
        client.delete_collection("bns_sections")
    except:
        pass
    
    collection = client.get_or_create_collection(
        name="bns_sections",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Process documents
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for chunk in tqdm(chunks, desc="Embedding"):
        # Ensure metadata is properly formatted
        metadata = {
            "section": str(chunk.get("section", "")),
            "section_display": f"Section {chunk.get('section', '')}",
            "heading": str(chunk.get("heading", "")),
            "chapter": str(chunk.get("chapter", "")),
            "length": int(chunk.get("length", 0))
        }
        
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["content"]],
            metadatas=[metadata],  # Note: wrapped in list
            embeddings=[model.encode(chunk["content"])]
        )
    
    print(f"✅ Successfully embedded {len(chunks)} sections")

if __name__ == "__main__":
    embed_chunks()
