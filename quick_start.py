# quick_start.py
import os
import sys
from dotenv import load_dotenv

def main():
    print("🚀 LLM-Enhanced Knowledge Graph Quick Start")
    print("=" * 50)
    
    # Load environment
    if not os.path.exists('.env'):
        print("❌ .env file not found. Please create it with your credentials.")
        return
    
    load_dotenv()
    
    # Check required files
    required_files = ['llm_graph_builder.py', 'enhanced_query_system.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ {file} not found. Please create this file.")
            return
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_start.py build    # Build the knowledge graph")
        print("  python quick_start.py query    # Run interactive queries")
        print("  python quick_start.py check    # Check setup")
        print("  python quick_start.py validate # Validate built graph")
        return
    
    command = sys.argv[1]
    
    if command == "check":
        # Run setup check
        check_setup()
        
    elif command == "build":
        # Build the graph
        docs_dir = "shiny_docs/content"
        structure_file = "shiny_docs/doc_structure.json"
        
        if not os.path.exists(docs_dir):
            print(f"❌ Documents directory '{docs_dir}' not found")
            return
        
        if not os.path.exists(structure_file):
            print(f"❌ Structure file '{structure_file}' not found")
            return
        
        print("🔨 Building LLM-Enhanced Knowledge Graph...")
        from llm_graph_builder import build_and_optimize_knowledge_graph
        
        success = build_and_optimize_knowledge_graph(docs_dir, structure_file)
        
        if success:
            print("✅ Graph built successfully!")
            print("🚀 Run queries with: python quick_start.py query")
        else:
            print("❌ Graph building failed. Check the errors above.")
            
    elif command == "query":
        # Run interactive queries
        print("🔍 Starting interactive query system...")
        from enhanced_query_system import EnhancedInteractiveCLI
        cli = EnhancedInteractiveCLI()
        cli.run()
        
    elif command == "validate":
        # Validate the graph
        print("🔍 Validating enhanced graph features...")
        from enhanced_query_system import validate_enhanced_graph
        validate_enhanced_graph()
        
    else:
        print(f"Unknown command: {command}")

def check_setup():
    """Check project setup"""
    print("🔍 Checking project setup...")
    
    all_good = True
    
    # Check required files
    required_files = {
        "llm_graph_builder.py": "LLM Graph Builder",
        "enhanced_query_system.py": "Enhanced Query System", 
        ".env": "Environment Variables",
        "requirements.txt": "Requirements File"
    }
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ {description}: {filename}")
        else:
            print(f"❌ Missing {description}: {filename}")
            all_good = False
    
    # Check if docs directory exists
    if os.path.exists("shiny_docs/content") and os.path.isdir("shiny_docs/content"):
        doc_count = len([f for f in os.listdir("shiny_docs") if f.endswith('.json')])
        print(f"✅ Documents directory: shiny_docs/content ({doc_count} JSON files)")
    else:
        print("❌ Missing documents directory: docs/")
        all_good = False
    
    # Check doc structure file
    if os.path.exists("shiny_docs/doc_structure.json"):
        print("✅ Document Structure: doc_structure.json")
    else:
        print("❌ Missing document structure: doc_structure.json")
        all_good = False
    
    if all_good:
        print("\n🎉 Project setup looks good!")
        print("\nNext steps:")
        print("1. Build the graph: python quick_start.py build")
        print("2. Run queries: python quick_start.py query")
    else:
        print("\n❌ Please fix the missing files above before proceeding.")
        
        print("\nTo create missing files:")
        if not os.path.exists("llm_graph_builder.py"):
            print("- Create llm_graph_builder.py with the LLM graph builder code")
        if not os.path.exists(".env"):
            print("- Create .env with: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY")
        if not os.path.exists("docs"):
            print("- Create docs/ directory with your JSON document files")
        if not os.path.exists("doc_structure.json"):
            print("- Create doc_structure.json with your document hierarchy")

if __name__ == "__main__":
    main()