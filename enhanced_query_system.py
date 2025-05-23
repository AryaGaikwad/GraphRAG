# Enhanced Query System with LLM-Built Knowledge Graph
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langgraph as lg
from langgraph.graph import StateGraph
from neo4j import GraphDatabase
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EnhancedShinyQuerySystem:
    """Enhanced query system that leverages LLM-built knowledge graph"""
    
    def __init__(self):
        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Create Neo4j driver and graph
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        self.graph = Neo4jGraph(
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            database="neo4j"
        )
        
        # LLM setup
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Setup enhanced prompts
        self._setup_enhanced_prompts()
        
        logger.info("Enhanced Shiny Query System initialized")
    
    def _setup_enhanced_prompts(self):
        """Setup enhanced prompts that leverage LLM-extracted entities and relationships"""
        
        # Enhanced query analysis prompt
        self.enhanced_query_analyzer_prompt = ChatPromptTemplate.from_template("""
        You are an expert in analyzing Shiny for Python queries and generating optimal Cypher queries.
        You have access to a knowledge graph built with LLM-enhanced entity extraction and relationship inference.
        
        GRAPH SCHEMA:
        {schema}
        
        AVAILABLE ENTITY TYPES:
        - Document: Documentation pages (title, url, description, llm_analysis)
        - Chunk: Content segments (text, type, embedding)
        - Component: UI components extracted by LLM (name, description)
        - Function: Functions/decorators extracted by LLM (name, description, parameters)
        - Topic: Key concepts identified by LLM (name, description)
        - CodeExample: Code snippets (code)
        - Cluster: Semantic clusters created by LLM (name, description, use_cases)
        
        ENHANCED RELATIONSHIP TYPES:
        - PART_OF: Chunks belong to documents
        - MENTIONED_IN: Entities mentioned in chunks
        - SIMILAR: Content similarity (with confidence scores)
        - EXAMPLE_OF: Code examples related to documents
        - DOCUMENTED_IN: Entities documented in pages (with frequency)
        - RELATES_TO: Topics related to components (LLM inferred)
        - IMPLEMENTS: Components implementing functions (LLM inferred)
        - USES: One entity uses another (LLM inferred)
        - DEPENDS_ON: Dependency relationships (LLM inferred)
        - WORKS_WITH: Entities that work well together (LLM inferred)
        - BELONGS_TO: Entities belonging to semantic clusters
        - ENABLES: One entity enables another (LLM inferred)
        
        USER QUERY: "{query}"
        
        Generate a comprehensive Cypher query that:
        1. Leverages LLM-extracted entities and relationships
        2. Uses semantic clusters when relevant
        3. Considers LLM confidence scores in relationships
        4. Searches across multiple entity types
        5. Returns rich, contextual information
        
        QUERY STRATEGIES:
        - Use UNION for multiple search approaches
        - Leverage semantic clusters for conceptual queries
        - Use relationship confidence scores for ranking
        - Include LLM analysis from documents when available
        - Consider workflow and dependency relationships
        
        Return ONLY the Cypher query without explanations.
        """)
        
        # Enhanced response generation prompt
        self.enhanced_response_prompt = ChatPromptTemplate.from_template("""
        You are an expert Shiny for Python assistant with access to LLM-enhanced knowledge graph results.
        
        USER QUERY: "{query}"
        
        KNOWLEDGE GRAPH RESULTS:
        {graph_result}
        
        {code_example_section}
        
        {cluster_context}
        
        {llm_analysis_context}
        
        Generate a comprehensive response that:
        1. **Directly answers** the user's question
        2. **Leverages LLM insights** from the knowledge graph
        3. **Provides context** from semantic clusters and relationships
        4. **Includes relevant code examples** with explanations
        5. **Suggests related concepts** based on graph relationships
        6. **Explains workflows** when applicable
        
        Format your response with:
        - Clear headings and structure
        - Code examples with explanations
        - Links to related concepts
        - Best practices when relevant
        
        End with 3 specific follow-up questions based on the graph relationships.
        """)
    
    def get_enhanced_schema(self):
        """Get comprehensive schema including LLM-enhanced elements"""
        try:
            with self.neo4j_driver.session() as session:
                # Get basic schema information
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]
                
                rel_types_result = session.run("CALL db.relationshipTypes()")
                relationship_types = [record["relationshipType"] for record in rel_types_result]
                
                # Get cluster information
                clusters_result = session.run("""
                MATCH (c:Cluster)
                RETURN c.name as cluster_name, c.description as description
                LIMIT 10
                """)
                clusters = [(record["cluster_name"], record["description"]) for record in clusters_result]
                
                # Get sample LLM-inferred relationships
                llm_rels_result = session.run("""
                MATCH ()-[r]->()
                WHERE r.source = 'llm_inference' AND r.confidence > 0.7
                RETURN type(r) as rel_type, r.confidence as confidence
                LIMIT 10
                """)
                llm_relationships = [(record["rel_type"], record["confidence"]) for record in llm_rels_result]
                
                schema = f"""
                NODE LABELS: {', '.join(labels)}
                
                RELATIONSHIP TYPES: {', '.join(relationship_types)}
                
                SEMANTIC CLUSTERS: {', '.join([f"{name} ({desc})" for name, desc in clusters])}
                
                LLM-INFERRED RELATIONSHIPS: {', '.join([f"{rel} (conf: {conf:.2f})" for rel, conf in llm_relationships])}
                
                ENHANCED FEATURES:
                - LLM-extracted entities with descriptions
                - Confidence-scored relationships
                - Semantic clustering
                - Document-level LLM analysis
                - Cross-document relationship inference
                """
                
                return schema
        except Exception as e:
            logger.error(f"Error getting enhanced schema: {e}")
            return "Enhanced schema unavailable"
    
    def enhanced_query_analyzer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced query analysis using LLM-built graph knowledge"""
        try:
            schema = self.get_enhanced_schema()
            
            chain = (
                {"schema": lambda _: schema, "query": lambda state: state["query"]}
                | self.enhanced_query_analyzer_prompt
                | self.llm
                | StrOutputParser()
            )
            
            raw_cypher = chain.invoke(state)
            
            # Clean the cypher query
            import re
            cypher = raw_cypher
            if "```" in raw_cypher:
                code_pattern = r"```(?:cypher)?\s*([\s\S]*?)```"
                code_match = re.search(code_pattern, raw_cypher)
                if code_match:
                    cypher = code_match.group(1).strip()
            
            logger.info(f"Generated enhanced Cypher query: {cypher}")
            return {**state, "cypher": cypher}
            
        except Exception as e:
            logger.error(f"Error in enhanced query analysis: {e}")
            return {**state, "cypher": "MATCH (n) RETURN n LIMIT 5"}
    
    def enhanced_graph_querier(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced queries with fallback strategies"""
        if not state.get("cypher"):
            return {**state, "graph_result": [{"error": "No Cypher query provided"}]}
        
        try:
            cypher = state["cypher"]
            if "LIMIT" not in cypher.upper():
                cypher = cypher + " LIMIT 15"
            
            logger.info(f"Executing enhanced query: {cypher}")
            
            # Execute primary query
            result = self.graph.query(cypher)
            
            # If no results, try enhanced fallback strategies
            if len(result) == 0:
                logger.info("No results from primary query, trying enhanced fallbacks...")
                result = self._execute_enhanced_fallbacks(state["query"])
            
            # Extract additional context
            context_info = self._extract_context_info(result)
            
            logger.info(f"Enhanced query returned {len(result)} results with context")
            return {
                **state, 
                "graph_result": result[:10],  # Limit results
                "context_info": context_info
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced graph querier: {e}")
            return {**state, "graph_result": [{"error": str(e)}]}
    
    def _execute_enhanced_fallbacks(self, query):
        """Enhanced fallback strategies using LLM-built graph features"""
        fallback_results = []
        
        search_terms = self._extract_search_terms(query)
        
        # Fallback 1: Semantic cluster search
        try:
            cluster_query = f"""
            MATCH (c:Cluster)
            WHERE {' OR '.join([f"toLower(c.name) CONTAINS toLower('{term}') OR toLower(c.description) CONTAINS toLower('{term}')" for term in search_terms])}
            MATCH (e)-[:BELONGS_TO]->(c)
            MATCH (e)-[:MENTIONED_IN]->(chunk:Chunk)
            RETURN c.name as cluster_name, e.name as entity_name, chunk.text as text
            LIMIT 5
            """
            fallback_results.extend(self.graph.query(cluster_query))
        except Exception as e:
            logger.warning(f"Cluster fallback failed: {e}")
        
        # Fallback 2: LLM-inferred relationship search
        try:
            llm_rel_query = f"""
            MATCH (source)-[r]->(target)
            WHERE r.source = 'llm_inference' AND r.confidence > 0.7
            AND ({' OR '.join([f"toLower(source.name) CONTAINS toLower('{term}') OR toLower(target.name) CONTAINS toLower('{term}')" for term in search_terms])})
            MATCH (source)-[:MENTIONED_IN]->(chunk:Chunk)
            RETURN source.name as source_entity, type(r) as relationship, target.name as target_entity, 
                   r.confidence as confidence, chunk.text as context
            LIMIT 5
            """
            fallback_results.extend(self.graph.query(llm_rel_query))
        except Exception as e:
            logger.warning(f"LLM relationship fallback failed: {e}")
        
        # Fallback 3: Enhanced text search with LLM analysis
        try:
            enhanced_text_query = f"""
            MATCH (d:Document)
            WHERE d.llm_analysis IS NOT NULL AND 
                  ({' OR '.join([f"toLower(d.llm_analysis) CONTAINS toLower('{term}')" for term in search_terms])})
            MATCH (c:Chunk)-[:PART_OF]->(d)
            RETURN d.title as title, d.llm_analysis as analysis, c.text as chunk_text
            LIMIT 5
            """
            fallback_results.extend(self.graph.query(enhanced_text_query))
        except Exception as e:
            logger.warning(f"Enhanced text search fallback failed: {e}")
        
        return fallback_results
    
    def _extract_search_terms(self, query):
        """Extract search terms from query"""
        stop_words = {'in', 'the', 'a', 'an', 'and', 'or', 'for', 'to', 'with', 'how', 'what', 'when', 'where', 'why', 'which', 'who', 'shiny'}
        words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
        return [word for word in words if word not in stop_words and len(word) > 2][:5]
    
    def _extract_context_info(self, results):
        """Extract additional context from query results"""
        context = {
            "clusters": set(),
            "relationships": set(),
            "entities": set()
        }
        
        for result in results:
            if "cluster_name" in result:
                context["clusters"].add(result["cluster_name"])
            if "relationship" in result:
                context["relationships"].add(result["relationship"])
            if "entity_name" in result:
                context["entities"].add(result["entity_name"])
            if "source_entity" in result and "target_entity" in result:
                context["entities"].add(result["source_entity"])
                context["entities"].add(result["target_entity"])
        
        return {k: list(v) for k, v in context.items()}
    
    def enhanced_response_generator(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced responses using LLM-built graph insights"""
        try:
            # Prepare context sections
            code_example_section = ""
            cluster_context = ""
            llm_analysis_context = ""
            
            # Extract code examples
            for item in state.get("graph_result", []):
                if "code" in item and isinstance(item["code"], str) and len(item["code"]) > 10:
                    code_example_section = f"RELEVANT CODE EXAMPLE:\n```python\n{item['code']}\n```"
                    break
            
            # Extract cluster context
            context_info = state.get("context_info", {})
            if context_info.get("clusters"):
                cluster_context = f"RELATED SEMANTIC CLUSTERS: {', '.join(context_info['clusters'])}"
            
            # Extract LLM analysis context
            for item in state.get("graph_result", []):
                if "analysis" in item and item["analysis"]:
                    llm_analysis_context = f"LLM DOCUMENT ANALYSIS:\n{item['analysis'][:500]}..."
                    break
            
            # Generate enhanced response
            chain = (
                {
                    "query": lambda state: state["query"],
                    "graph_result": lambda state: json.dumps(state.get("graph_result", []), indent=2),
                    "code_example_section": lambda _: code_example_section,
                    "cluster_context": lambda _: cluster_context,
                    "llm_analysis_context": lambda _: llm_analysis_context
                }
                | self.enhanced_response_prompt
                | self.llm
            )
            
            response = chain.invoke(state)
            content = response.content
            
            # Generate contextual follow-up questions
            follow_up_questions = self._generate_contextual_followups(state)
            
            logger.info("Enhanced response generated with contextual insights")
            return {**state, "final_answer": content, "follow_up_questions": follow_up_questions}
            
        except Exception as e:
            logger.error(f"Error in enhanced response generation: {e}")
            return {**state, "final_answer": "Error generating response", "follow_up_questions": []}
    
    def _generate_contextual_followups(self, state):
        """Generate follow-up questions based on graph context"""
        try:
            context_info = state.get("context_info", {})
            base_questions = []
            
            # Questions based on clusters
            if context_info.get("clusters"):
                cluster = context_info["clusters"][0]
                base_questions.append(f"What other concepts are related to {cluster}?")
            
            # Questions based on relationships
            if context_info.get("relationships"):
                rel = context_info["relationships"][0]
                base_questions.append(f"How do I work with components that have {rel.lower()} relationships?")
            
            # Questions based on entities
            if context_info.get("entities"):
                entity = context_info["entities"][0]
                base_questions.append(f"Can you show me more examples using {entity}?")
            
            # Default questions if no context
            if not base_questions:
                base_questions = [
                    "How do I get started with Shiny for Python?",
                    "What are the key components I need to know?",
                    "Can you show me a complete example?"
                ]
            
            return base_questions[:3]
            
        except Exception as e:
            logger.error(f"Error generating contextual follow-ups: {e}")
            return ["What else would you like to know about Shiny?"]
    
    def build_enhanced_workflow(self):
        """Build the enhanced query workflow"""
        workflow = StateGraph(Dict)
        
        # Add enhanced nodes
        workflow.add_node("analyze_query", self.enhanced_query_analyzer)
        workflow.add_node("query_graph", self.enhanced_graph_querier)
        workflow.add_node("generate_response", self.enhanced_response_generator)
        
        # Add edges
        workflow.add_edge("analyze_query", "query_graph")
        workflow.add_edge("query_graph", "generate_response")
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        return workflow.compile()
    
    def query_enhanced_shiny_docs(self, user_query: str):
        """Query the enhanced Shiny documentation system"""
        logger.info(f"Processing enhanced query: {user_query}")
        
        workflow = self.build_enhanced_workflow()
        initial_state = {"query": user_query}
        
        try:
            final_state = workflow.invoke(initial_state)
            
            print("\n" + "="*60)
            print("🧠 ENHANCED SHINY DOCUMENTATION ASSISTANT")
            print("="*60)
            print(f"\n🔍 Query: {user_query}\n")
            
            # Display context information
            context_info = final_state.get("context_info", {})
            if any(context_info.values()):
                print("🏷️  Context:")
                if context_info.get("clusters"):
                    print(f"   Clusters: {', '.join(context_info['clusters'])}")
                if context_info.get("entities"):
                    print(f"   Entities: {', '.join(context_info['entities'][:3])}")
                if context_info.get("relationships"):
                    print(f"   Relationships: {', '.join(context_info['relationships'][:3])}")
                print()
            
            print("📝 Answer:\n")
            answer = final_state.get("final_answer", "No answer generated.")
            print(answer)
            print("\n" + "-"*60)
            
            follow_up_questions = final_state.get("follow_up_questions", [])
            if follow_up_questions:
                print("🔄 Contextual follow-up questions:")
                for i, question in enumerate(follow_up_questions, 1):
                    print(f"{i}. {question}")
            
            print("="*60)
            return final_state
            
        except Exception as e:
            logger.error(f"Error processing enhanced query: {e}")
            return {"query": user_query, "error": str(e)}
    
    def get_graph_statistics(self):
        """Get statistics about the enhanced knowledge graph"""
        try:
            with self.neo4j_driver.session() as session:
                stats = session.run("""
                RETURN 
                    size((n:Document)) as documents,
                    size((n:Chunk)) as chunks,
                    size((n:Component)) as components,
                    size((n:Function)) as functions,
                    size((n:Topic)) as topics,
                    size((n:Cluster)) as clusters,
                    size((n:CodeExample)) as code_examples,
                    size(()-[r {source: 'llm_inference'}]->()) as llm_relationships,
                    size(()-[]->()) as total_relationships
                """).single()
                
                return dict(stats)
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {}
    
    def cleanup(self):
        """Clean up connections"""
        try:
            self.neo4j_driver.close()
            logger.info("Neo4j connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# Interactive CLI for the enhanced system
class EnhancedInteractiveCLI:
    """Enhanced interactive command line interface"""
    
    def __init__(self):
        self.query_system = EnhancedShinyQuerySystem()
        self.show_welcome()
    
    def show_welcome(self):
        """Display welcome message with graph statistics"""
        stats = self.query_system.get_graph_statistics()
        
        print("\n" + "="*70)
        print("🧠 ENHANCED SHINY DOCUMENTATION ASSISTANT")
        print("   Powered by LLM-Built Knowledge Graph")
        print("="*70)
        
        if stats:
            print("\n📊 Knowledge Graph Statistics:")
            print(f"   Documents: {stats.get('documents', 0)}")
            print(f"   Content Chunks: {stats.get('chunks', 0)}")
            print(f"   LLM-Extracted Components: {stats.get('components', 0)}")
            print(f"   LLM-Extracted Functions: {stats.get('functions', 0)}")
            print(f"   LLM-Identified Topics: {stats.get('topics', 0)}")
            print(f"   Semantic Clusters: {stats.get('clusters', 0)}")
            print(f"   Code Examples: {stats.get('code_examples', 0)}")
            print(f"   LLM-Inferred Relationships: {stats.get('llm_relationships', 0)}")
            print(f"   Total Relationships: {stats.get('total_relationships', 0)}")
        
        print("\n🚀 Features:")
        print("   • LLM-enhanced entity extraction")
        print("   • Semantic clustering")
        print("   • Confidence-scored relationships")
        print("   • Cross-document relationship inference")
        print("   • Contextual response generation")
        
        print("\n💡 Try asking about:")
        print("   • 'How do I create interactive inputs?'")
        print("   • 'Show me examples of reactivity'")
        print("   • 'What components work well together?'")
        print("   • 'How do I build a dashboard layout?'")
        
        print("\nType 'exit' to quit, 'stats' for graph statistics.\n")
    
    def run(self):
        """Run the interactive CLI"""
        try:
            while True:
                user_input = input("🔍 Your question: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                    continue
                elif not user_input:
                    continue
                
                # Process the query
                result = self.query_system.query_enhanced_shiny_docs(user_input)
                
                # Brief pause for better UX
                print()
                
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
        except Exception as e:
            logger.error(f"CLI error: {e}")
            print(f"❌ An error occurred: {e}")
        finally:
            self.query_system.cleanup()
    
    def show_statistics(self):
        """Show current graph statistics"""
        stats = self.query_system.get_graph_statistics()
        print("\n📊 Current Knowledge Graph Statistics:")
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            print(f"   {formatted_key}: {value}")
        print()


# Main execution
if __name__ == "__main__":
    try:
        # Check if we're building the graph or running queries
        import sys
        
        if len(sys.argv) > 1 and sys.argv[1] == "build":
            # Build the enhanced knowledge graph
            from llm_graph_builder import build_and_optimize_knowledge_graph
            
            if len(sys.argv) < 4:
                print("Usage: python enhanced_query_system.py build <docs_dir> <doc_structure_path>")
                sys.exit(1)
            
            docs_dir = sys.argv[2]
            doc_structure_path = sys.argv[3]
            
            print("🔨 Building LLM-Enhanced Knowledge Graph...")
            success = build_and_optimize_knowledge_graph(docs_dir, doc_structure_path)
            
            if success:
                print("✅ Knowledge graph built successfully!")
                print("🚀 You can now run queries with: python enhanced_query_system.py")
            else:
                print("❌ Failed to build knowledge graph")
                sys.exit(1)
        
        else:
            # Run the interactive query system
            cli = EnhancedInteractiveCLI()
            cli.run()
            
    except Exception as e:
        logger.error(f"Main execution error: {e}")
        print(f"❌ Error: {e}")


# Utility functions for integration
def migrate_from_basic_to_enhanced():
    """Utility to help migrate from basic to enhanced system"""
    print("""
    🔄 MIGRATION GUIDE: Basic to Enhanced System
    
    1. Build Enhanced Graph:
       python enhanced_query_system.py build /path/to/docs /path/to/structure.json
    
    2. Run Enhanced Queries:
       python enhanced_query_system.py
    
    Key Improvements:
    ✅ LLM extracts entities with descriptions
    ✅ Confidence-scored relationships  
    ✅ Semantic clustering
    ✅ Cross-document relationship inference
    ✅ Enhanced fallback strategies
    ✅ Contextual response generation
    """)

def validate_enhanced_graph():
    """Validate that the enhanced graph has expected LLM features"""
    query_system = EnhancedShinyQuerySystem()
    stats = query_system.get_graph_statistics()
    
    validation_results = {
        "has_llm_relationships": stats.get('llm_relationships', 0) > 0,
        "has_clusters": stats.get('clusters', 0) > 0,
        "has_components": stats.get('components', 0) > 0,
        "has_functions": stats.get('functions', 0) > 0,
        "has_topics": stats.get('topics', 0) > 0
    }
    
    query_system.cleanup()
    
    print("\n🔍 Enhanced Graph Validation:")
    for feature, exists in validation_results.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")
    
    all_good = all(validation_results.values())
    print(f"\n{'✅ Enhanced graph is ready!' if all_good else '❌ Graph needs enhancement'}")
    
    return all_good