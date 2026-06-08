import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agent_router import classify_query, detect_acts_from_query
from app.scenario_processor import analyze_scenario
from app.retriever import retrieve_direct, retrieve_section

def test_enhanced_system():
    """Test the enhanced multi-act system"""
    
    test_queries = [
        "Explain Section 302 of BNS",
        "What is theft under IPC",
        "A mob vandalized shops during protest",
        "How to prove murder with evidence",
        "Procedure for filing civil suit"
    ]
    
    print("🧪 Testing Enhanced Multi-Act System\n")
    
    for query in test_queries:
        print(f"🔍 Query: '{query}'")
        
        # Classify query
        query_type = classify_query(query)
        print(f"   Type: {query_type}")
        
        # Detect relevant acts
        acts = detect_acts_from_query(query)
        print(f"   Relevant acts: {acts}")
        
        # Test retrieval based on type
        if query_type == "section":
            # Extract section number
            import re
            section_match = re.search(r'(\d+)', query)
            if section_match:
                results = retrieve_section(section_match.group(1))
                print(f"   Section found in: {list(results.keys())}")
        
        elif query_type == "direct":
            results = retrieve_direct(query, n_results=2)
            print(f"   Retrieved {len(results['documents'])} documents")
        
        elif query_type == "scenario":
            analysis = analyze_scenario(query)
            print(f"   Primary offense: {analysis['primary_offense']}")
            print(f"   Suggested acts: {analysis['relevant_acts']}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_system()