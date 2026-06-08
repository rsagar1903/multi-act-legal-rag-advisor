import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import asyncio
import concurrent.futures
from sentence_transformers import SentenceTransformer

class MultiCollectionManager:
    def __init__(self, db_path: str = "multi_act_db"):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.act_priority = ["BNS", "IPC", "CRPC", "CPC", "BSA"]  # Priority order
        
    def list_available_acts(self) -> List[str]:
        """List all available legal acts in the database"""
        collections = self.client.list_collections()
        return [coll.name for coll in collections]
    
    def get_collection(self, act_name: str):
        """Get a specific collection by act name"""
        try:
            return self.client.get_collection(act_name)
        except:
            return None
    
    def query_all_acts(self, query_text: str, n_results: int = 3) -> Dict[str, List]:
        """
        Query all legal acts in parallel and return aggregated results
        Returns: {act_name: {"documents": [], "metadatas": []}}
        """
        collections = self.client.list_collections()
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all queries in parallel
            future_to_act = {
                executor.submit(
                    self._query_single_act,
                    coll.name,
                    query_text,
                    n_results
                ): coll.name for coll in collections
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_act):
                act_name = future_to_act[future]
                try:
                    results[act_name] = future.result()
                except Exception as e:
                    print(f"Error querying {act_name}: {str(e)}")
                    results[act_name] = {"documents": [], "metadatas": []}
        
        return results
    
    def _query_single_act(self, act_name: str, query_text: str, n_results: int) -> Dict:
        """Query a single legal act"""
        collection = self.get_collection(act_name)
        if not collection:
            return {"documents": [], "metadatas": []}
        
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            metadatas = results["metadatas"][0]
            distances = results.get("distances", [[]])[0]
            for index, meta in enumerate(metadatas):
                if not isinstance(meta, dict):
                    continue
                if index < len(distances):
                    try:
                        meta["confidence"] = round(max(0.0, min(1.0, 1.0 - float(distances[index]))), 4)
                    except (TypeError, ValueError):
                        pass
            return {
                "documents": results["documents"][0],
                "metadatas": metadatas
            }
        except Exception as e:
            print(f"Query failed for {act_name}: {str(e)}")
            return {"documents": [], "metadatas": []}
    
    def query_specific_acts(self, act_names: List[str], query_text: str, n_results: int = 3) -> Dict[str, List]:
        """Query only specific legal acts"""
        results = {}
        for act_name in act_names:
            results[act_name] = self._query_single_act(act_name, query_text, n_results)
        return results
    
    def get_section_across_acts(self, section_number: str) -> Dict[str, List]:
        """
        Find a specific section number across all legal acts
        Returns results from all acts that have this section
        """
        collections = self.client.list_collections()
        results = {}
        
        for coll in collections:
            try:
                exact_results = coll.get(
                    where={"section": section_number},
                    include=["documents", "metadatas"]
                )
                if exact_results["documents"]:
                    results[coll.name] = {
                        "documents": exact_results["documents"],
                        "metadatas": exact_results["metadatas"]
                    }
            except Exception as e:
                print(f"Error getting section {section_number} from {coll.name}: {str(e)}")
        
        return results

# Singleton instance for easy access
multi_collection_manager = MultiCollectionManager()

# Test function
def test_multi_collection():
    """Test the multi-collection manager"""
    manager = MultiCollectionManager()
    
    print("📋 Available acts:", manager.list_available_acts())
    
    # Test 1: Query across all acts
    print("\n🔍 Testing cross-act query...")
    results = manager.query_all_acts("theft", n_results=2)
    for act, data in results.items():
        print(f"  {act}: {len(data['documents'])} results")
    
    # Test 2: Specific section search
    print("\n🔍 Testing section search...")
    section_results = manager.get_section_across_acts("302")
    for act, data in section_results.items():
        print(f"  Section 302 found in {act}: {len(data['documents'])} matches")

if __name__ == "__main__":
    test_multi_collection()
