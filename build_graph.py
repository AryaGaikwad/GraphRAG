# build_graph.py
import os
import sys
from dotenv import load_dotenv
from llm_graph_builder import LLMEnhancedGraphBuilder, AdvancedLLMGraphOperations

# Load environment variables
load_dotenv()

def build_enhanced_knowledge_graph(docs_dir, doc_structure_path):
    """Build the LLM-enhanced knowledge graph"""
    
    # Neo4j connection details
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    print("🚀 Starting LLM-Enhanced Knowledge Graph Construction...")
    
    # Initialize the graph builder
    builder = LLMEnhancedGraphBuilder(
        uri=neo4j_uri,
        auth=(neo4j_user, neo4j_password)
    )
    
    try:
        # Phase 1: Build the basic enhanced graph
        print("📊 Phase 1: Building LLM-Enhanced Knowledge Graph...")
        builder.build_llm_enhanced_graph(docs_dir, doc_structure_path)
        
        # Phase 2: Advanced optimization
        print("🔧 Phase 2: Advanced Graph Optimization...")
        optimizer = AdvancedLLMGraphOperations(builder)
        optimization_report = optimizer.optimize_graph_structure()
        
        print("✅ Graph building completed successfully!")
        print("\n📋 Optimization Report:")
        print(f"   Missing relationships added: {optimization_report.get('missing_relationships_added', 0)}")
        print(f"   Semantic clusters created: {optimization_report.get('semantic_clusters_created', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        builder.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_graph.py <docs_directory> <doc_structure_file>")
        print("Example: python build_graph.py ./docs ./doc_structure.json")
        sys.exit(1)
    
    docs_dir = sys.argv[1]
    doc_structure_path = sys.argv[2]
    
    # Validate paths exist
    if not os.path.exists(docs_dir):
        print(f"❌ Error: Documents directory '{docs_dir}' not found")
        sys.exit(1)
    
    if not os.path.exists(doc_structure_path):
        print(f"❌ Error: Document structure file '{doc_structure_path}' not found")
        sys.exit(1)
    
    print(f"📁 Documents directory: {docs_dir}")
    print(f"📄 Structure file: {doc_structure_path}")
    
    success = build_enhanced_knowledge_graph(docs_dir, doc_structure_path)
    
    if success:
        print("\n🎉 SUCCESS! Your LLM-enhanced knowledge graph is ready!")
        print("🚀 You can now run queries with:")
        print("   python enhanced_query_system.py")
    else:
        print("\n❌ FAILED! Please check the errors above and try again.")