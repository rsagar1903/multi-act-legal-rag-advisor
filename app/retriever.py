from tqdm import tqdm
try:
    from .multi_collection_manager import MultiCollectionManager
    from .agent_router import detect_acts_from_query
except ImportError:
    from multi_collection_manager import MultiCollectionManager
    from agent_router import detect_acts_from_query
import re

manager = MultiCollectionManager()

def retrieve_parallel(concepts, collection=None, query_text=""):
    """
    Enhanced retriever that works across multiple acts
    concepts: List of legal concepts to search for
    query_text: Original query for act detection
    """
    # Detect which acts to search
    acts_to_search = detect_acts_from_query(query_text)
    act_collections = [f"{act}_sections" for act in acts_to_search]
    
    results = {"documents": [], "metadatas": []}
    
    for concept in tqdm(concepts, desc="Multi-act search"):
        # Check if concept is a section number
        if re.match(r'^\d+$', str(concept)):
            # Search for section across detected acts
            section_results = manager.get_section_across_acts(concept)
            for act_data in section_results.values():
                results["documents"].extend(act_data["documents"])
                results["metadatas"].extend(act_data["metadatas"])
        else:
            # Semantic search in detected acts
            for act_collection in act_collections:
                try:
                    vector_results = manager._query_single_act(act_collection, concept, 2)
                    results["documents"].extend(vector_results["documents"])
                    results["metadatas"].extend(vector_results["metadatas"])
                except Exception as e:
                    print(f"Error searching {act_collection}: {str(e)}")
    
    return results

def retrieve_direct(query_text: str, n_results: int = 5):
    """Direct query across relevant acts"""
    acts_to_search = detect_acts_from_query(query_text)
    act_collections = [f"{act}_sections" for act in acts_to_search]
    
    results = {"documents": [], "metadatas": []}
    
    for act_collection in act_collections:
        try:
            vector_results = manager._query_single_act(act_collection, query_text, n_results)
            results["documents"].extend(vector_results["documents"])
            results["metadatas"].extend(vector_results["metadatas"])
        except Exception as e:
            print(f"Error searching {act_collection}: {str(e)}")
    
    return results

def retrieve_section(section_number: str):
    """Retrieve specific section across all acts"""
    return manager.get_section_across_acts(section_number)
