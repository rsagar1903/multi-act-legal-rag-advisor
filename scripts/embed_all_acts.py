import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import os

# Configuration
ACTS_CONFIG = {
    "bns": {
        "chunk_file": "data/bns_chunks.json",
        "collection_name": "bns_sections"
    },
    "ipc": {
        "chunk_file": "data/ipc_chunks.json", 
        "collection_name": "ipc_sections"
    },
    "crpc": {
        "chunk_file": "data/crpc_chunks.json",
        "collection_name": "crpc_sections"
    },
    "cpc": {
        "chunk_file": "data/cpc_chunks.json",
        "collection_name": "cpc_sections"
    },
    "bsa": {
        "chunk_file": "data/bsa_chunks.json",
        "collection_name": "bsa_sections"
    }
}

def embed_act(act_name, config, client, model):
    """Embed a single legal act"""
    print(f"📚 Embedding {act_name.upper()}...")
    
    try:
        with open(config["chunk_file"], "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Skipping {act_name}: File not found")
        return
    
    # Clear existing collection
    try:
        client.delete_collection(config["collection_name"])
    except:
        pass
    
    collection = client.get_or_create_collection(
        name=config["collection_name"],
        metadata={"hnsw:space": "cosine", "act": act_name}
    )
    
    # Process chunks
    for chunk in tqdm(chunks, desc=f"Embedding {act_name}"):
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["content"]],
            metadatas=[{
                "section": chunk["section"],
                "section_display": chunk.get("section_display", f"Section {chunk['section']}"),
                "heading": chunk["heading"],
                "chapter": chunk.get("chapter", ""),
                "act": act_name.upper()
            }],
            embeddings=[model.encode(chunk["content"])]
        )
    
    print(f"✅ {act_name.upper()}: {collection.count()} sections embedded")

def main():
    """Embed all legal acts into separate collections"""
    client = chromadb.PersistentClient(
        path="multi_act_db",
        settings=Settings(anonymized_telemetry=False)
    )
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🚀 Starting multi-act embedding process...")
    for act_name, config in ACTS_CONFIG.items():
        embed_act(act_name, config, client, model)
    
    print("🎉 All acts embedded successfully!")
    print("\nCollections created:")
    for act_name in ACTS_CONFIG.keys():
        print(f"• {act_name}_sections")

if __name__ == "__main__":
    main()