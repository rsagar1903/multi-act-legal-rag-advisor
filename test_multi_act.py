from app.multi_collection_manager import MultiCollectionManager

def interactive_test():
    """Interactive test of the multi-collection system"""
    manager = MultiCollectionManager()
    
    print("🎯 Multi-Act Legal Database Test Interface")
    print("Available acts:", manager.list_available_acts())
    print("\nCommands:")
    print("  query [text] - Search across all acts")
    print("  section [number] - Find section across acts") 
    print("  acts [act1,act2] [query] - Search specific acts")
    print("  exit - Quit")
    
    while True:
        try:
            command = input("\n➡️  Enter command: ").strip()
            
            if command.lower() == 'exit':
                break
                
            elif command.startswith('query '):
                query_text = command[6:]
                print(f"🔍 Searching for: '{query_text}'")
                results = manager.query_all_acts(query_text, n_results=2)
                
                for act, data in results.items():
                    if data['documents']:
                        print(f"\n📚 {act}:")
                        for i, (doc, meta) in enumerate(zip(data['documents'], data['metadatas'])):
                            print(f"   {i+1}. {meta.get('section_display', 'Section')}: {meta.get('heading', '')}")
                            print(f"      {doc[:100]}...")
            
            elif command.startswith('section '):
                section_num = command[8:]
                print(f"🔍 Finding section {section_num}")
                results = manager.get_section_across_acts(section_num)
                
                if results:
                    for act, data in results.items():
                        print(f"\n📚 {act}:")
                        for i, (doc, meta) in enumerate(zip(data['documents'], data['metadatas'])):
                            print(f"   {meta.get('section_display', 'Section')}: {meta.get('heading', '')}")
                            print(f"      {doc[:200]}...")
                else:
                    print("❌ Section not found in any act")
            
            elif command.startswith('acts '):
                parts = command[5:].split(' ', 1)
                if len(parts) == 2:
                    act_names = [act.strip() for act in parts[0].split(',')]
                    query_text = parts[1]
                    print(f"🔍 Searching {act_names} for: '{query_text}'")
                    
                    results = manager.query_specific_acts(act_names, query_text, n_results=2)
                    for act, data in results.items():
                        if data['documents']:
                            print(f"\n📚 {act}:")
                            for i, (doc, meta) in enumerate(zip(data['documents'], data['metadatas'])):
                                print(f"   {i+1}. {meta.get('section_display', 'Section')}: {meta.get('heading', '')}")
            
            else:
                print("❌ Unknown command. Try: query, section, acts, or exit")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    interactive_test()