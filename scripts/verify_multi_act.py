import chromadb
from chromadb.config import Settings

def verify_collections():
    client = chromadb.PersistentClient(
        path="multi_act_db",
        settings=Settings()
    )
    
    collections = client.list_collections()
    print("📋 Available collections:")
    for coll in collections:
        count = client.get_collection(coll.name).count()
        print(f"• {coll.name}: {count} items")
        print(f"  Metadata: {coll.metadata}")

if __name__ == "__main__":
    verify_collections()