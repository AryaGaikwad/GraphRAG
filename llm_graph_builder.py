# LLM-Enhanced Knowledge Graph Builder
from neo4j import GraphDatabase
import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtraction(BaseModel):
    """Structured output for entity extraction"""
    components: List[Dict[str, str]] = Field(description="UI components with name and description")
    functions: List[Dict[str, str]] = Field(description="Functions with name, parameters, and description")
    topics: List[Dict[str, str]] = Field(description="Key topics with name and description")
    relationships: List[Dict[str, str]] = Field(description="Relationships between entities")

class ConceptualRelationship(BaseModel):
    """Structured output for conceptual relationships"""
    source_entity: str = Field(description="Source entity name")
    relationship_type: str = Field(description="Type of relationship")
    target_entity: str = Field(description="Target entity name")
    confidence: float = Field(description="Confidence score 0-1")
    explanation: str = Field(description="Brief explanation of the relationship")

class LLMEnhancedGraphBuilder:
    def __init__(self, uri, auth, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.database = database
        
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Initialize embedding model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Create LLM prompts
        self._setup_prompts()
        
    def close(self):
        self.driver.close()
    
    def _setup_prompts(self):
        """Setup LLM prompts for various graph building tasks"""
        
        # Entity extraction prompt
        self.entity_extraction_prompt = ChatPromptTemplate.from_template("""
        You are an expert in analyzing Shiny for Python documentation to extract structured entities.
        
        Analyze the following text and extract:
        1. **Components**: UI elements like inputs, outputs, layouts (e.g., ui.input_text, ui.output_plot)
        2. **Functions**: Python functions, decorators, methods (e.g., @render.text, server functions)
        3. **Topics**: Key concepts and themes (e.g., reactivity, layouts, modules)
        4. **Relationships**: How entities relate to each other
        
        Text to analyze:
        {text}
        
        For each entity, provide:
        - name: The exact name/identifier
        - description: A brief description of what it does
        
        For relationships, specify:
        - source: Source entity name
        - relationship: Type of relationship (uses, implements, part_of, related_to)
        - target: Target entity name
        
        Return a structured response with clear categories.
        """)
        
        # Relationship inference prompt
        self.relationship_prompt = ChatPromptTemplate.from_template("""
        You are an expert in Shiny for Python architecture and relationships.
        
        Given these entities from a documentation chunk:
        Components: {components}
        Functions: {functions}
        Topics: {topics}
        
        And this context text:
        {context}
        
        Infer meaningful relationships between these entities. Consider:
        - Which components use which functions?
        - Which topics are related to which components/functions?
        - Which entities work together or have dependencies?
        - Which entities are part of larger concepts?
        
        For each relationship, provide:
        - source_entity: The source entity name
        - relationship_type: One of (uses, implements, part_of, related_to, depends_on, enables)
        - target_entity: The target entity name  
        - confidence: Score from 0.0 to 1.0
        - explanation: Brief explanation why this relationship exists
        
        Only include relationships with confidence > 0.7.
        """)
        
        # Document summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_template("""
        Analyze this Shiny documentation and provide a structured summary:
        
        Title: {title}
        Content: {content}
        
        Extract:
        1. **Main Purpose**: What is this document about?
        2. **Key Concepts**: What are the 3-5 most important concepts explained?
        3. **Code Patterns**: What coding patterns or examples are shown?
        4. **Dependencies**: What other Shiny concepts does this depend on?
        5. **Use Cases**: What problems does this solve or when would you use this?
        
        Provide a clear, structured response that will help in building knowledge graph relationships.
        """)
        
        # Chunk relationship prompt
        self.chunk_relationship_prompt = ChatPromptTemplate.from_template("""
        You are analyzing relationships between documentation chunks.
        
        Chunk 1: {chunk1}
        Chunk 2: {chunk2}
        
        Determine if these chunks are related and how:
        
        Relationship types to consider:
        - FOLLOWS: Chunk 2 logically follows chunk 1 (tutorial progression)
        - REFERENCES: One chunk references concepts from the other
        - SIMILAR: Both chunks discuss similar topics
        - PREREQUISITE: Chunk 1 is prerequisite knowledge for chunk 2
        - EXAMPLE_OF: One chunk provides examples for concepts in the other
        
        Respond with:
        - relationship_type: One of the above types or "NONE"
        - confidence: Score from 0.0 to 1.0
        - explanation: Brief explanation of the relationship
        
        Only suggest relationships with confidence > 0.6.
        """)
    
    def create_constraints(self):
        """Create necessary constraints in Neo4j"""
        with self.driver.session(database=self.database) as session:
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT component_name IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT function_name IF NOT EXISTS FOR (f:Function) REQUIRE f.name IS UNIQUE",
                "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Constraint creation warning: {e}")
            
            # Create vector index
            try:
                index_result = session.run("""
                SHOW INDEXES
                YIELD name, type
                WHERE name = 'chunk_embeddings' AND type = 'VECTOR'
                RETURN count(*) as count
                """)
                
                if index_result.single()["count"] == 0:
                    session.run("""
                    CALL db.index.vector.createNodeIndex(
                    'chunk_embeddings',
                    'Chunk',
                    'embedding',
                    768,
                    'cosine'
                    )
                    """)
                    logger.info("Vector index created successfully")
                else:
                    logger.info("Vector index already exists")
            except Exception as e:
                logger.warning(f"Vector index creation warning: {e}")
    
    def generate_embedding(self, text):
        """Generate embedding for text"""
        try:
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            return embedding[0].numpy().tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 768
    
    def llm_extract_entities(self, text, chunk_id):
        """Use LLM to extract entities from text"""
        try:
            # Get LLM analysis
            chain = self.entity_extraction_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": text})
            
            # Parse the result and create entities
            entities = self._parse_entity_response(result)
            
            # Create entities in Neo4j
            self._create_entities_in_neo4j(entities, chunk_id)
            
            return entities
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return {"components": [], "functions": [], "topics": []}
    
    def _parse_entity_response(self, response):
        """Parse LLM response to extract structured entities"""
        entities = {"components": [], "functions": [], "topics": []}
        
        try:
            # Simple parsing - look for patterns in the response
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect section headers
                if 'component' in line.lower() and ':' in line:
                    current_section = 'components'
                elif 'function' in line.lower() and ':' in line:
                    current_section = 'functions'
                elif 'topic' in line.lower() and ':' in line:
                    current_section = 'topics'
                elif line.startswith('-') or line.startswith('*'):
                    # Parse entity entries
                    if current_section and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            name = parts[0].strip('- *').strip()
                            description = parts[1].strip()
                            entities[current_section].append({
                                "name": name,
                                "description": description
                            })
            
            # Fallback: extract using regex if structured parsing fails
            if not any(entities.values()):
                entities = self._regex_extract_entities(response)
                
        except Exception as e:
            logger.error(f"Error parsing entity response: {e}")
            
        return entities
    
    def _regex_extract_entities(self, text):
        """Fallback regex-based entity extraction"""
        entities = {"components": [], "functions": [], "topics": []}
        
        # Extract UI components
        ui_components = re.findall(r'ui\.([a-zA-Z0-9_]+)', text)
        for comp in ui_components:
            entities["components"].append({"name": f"ui.{comp}", "description": f"Shiny UI component"})
        
        # Extract render functions
        render_functions = re.findall(r'@render\.([a-zA-Z0-9_]+)', text)
        for func in render_functions:
            entities["functions"].append({"name": f"@render.{func}", "description": f"Shiny render decorator"})
        
        # Extract key topics
        topics = ["reactivity", "express", "core", "layout", "module", "widget", "dashboard"]
        for topic in topics:
            if re.search(r'\b' + topic + r'\b', text.lower()):
                entities["topics"].append({"name": topic, "description": f"Shiny {topic} concept"})
        
        return entities
    
    def _create_entities_in_neo4j(self, entities, chunk_id):
        """Create extracted entities in Neo4j"""
        with self.driver.session(database=self.database) as session:
            # Create components
            for comp in entities["components"]:
                session.run("""
                MERGE (c:Component {name: $name})
                SET c.description = $description
                WITH c
                MATCH (chunk:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONED_IN]->(chunk)
                """, {
                    "name": comp["name"],
                    "description": comp["description"],
                    "chunk_id": chunk_id
                })
            
            # Create functions
            for func in entities["functions"]:
                session.run("""
                MERGE (f:Function {name: $name})
                SET f.description = $description
                WITH f
                MATCH (chunk:Chunk {id: $chunk_id})
                MERGE (f)-[:MENTIONED_IN]->(chunk)
                """, {
                    "name": func["name"],
                    "description": func["description"],
                    "chunk_id": chunk_id
                })
            
            # Create topics
            for topic in entities["topics"]:
                session.run("""
                MERGE (t:Topic {name: $name})
                SET t.description = $description
                WITH t
                MATCH (chunk:Chunk {id: $chunk_id})
                MERGE (t)-[:MENTIONED_IN]->(chunk)
                """, {
                    "name": topic["name"],
                    "description": topic["description"],
                    "chunk_id": chunk_id
                })
    
    def llm_infer_relationships(self, chunk_id, context_text):
        """Use LLM to infer relationships between entities"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get entities mentioned in this chunk
                result = session.run("""
                MATCH (e)-[:MENTIONED_IN]->(c:Chunk {id: $chunk_id})
                RETURN 
                    collect(CASE WHEN e:Component THEN e.name END) as components,
                    collect(CASE WHEN e:Function THEN e.name END) as functions,
                    collect(CASE WHEN e:Topic THEN e.name END) as topics
                """, {"chunk_id": chunk_id})
                
                record = result.single()
                if not record:
                    return
                
                components = [c for c in record["components"] if c]
                functions = [f for f in record["functions"] if f]
                topics = [t for t in record["topics"] if t]
                
                if not any([components, functions, topics]):
                    return
                
                # Use LLM to infer relationships
                chain = self.relationship_prompt | self.llm | StrOutputParser()
                relationship_result = chain.invoke({
                    "components": components,
                    "functions": functions,
                    "topics": topics,
                    "context": context_text[:2000]  # Limit context length
                })
                
                # Parse and create relationships
                relationships = self._parse_relationships(relationship_result)
                self._create_relationships_in_neo4j(relationships)
                
        except Exception as e:
            logger.error(f"Error inferring relationships: {e}")
    
    def _parse_relationships(self, response):
        """Parse LLM relationship response"""
        relationships = []
        
        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if '->' in line or 'uses' in line or 'implements' in line or 'relates' in line:
                    # Try to extract relationship components
                    parts = re.split(r'[->]|uses|implements|relates', line)
                    if len(parts) >= 2:
                        source = parts[0].strip()
                        target = parts[-1].strip()
                        rel_type = "RELATES_TO"  # Default relationship type
                        
                        if 'uses' in line:
                            rel_type = "USES"
                        elif 'implements' in line:
                            rel_type = "IMPLEMENTS"
                        elif 'part_of' in line:
                            rel_type = "PART_OF"
                        
                        relationships.append({
                            "source": source,
                            "relationship": rel_type,
                            "target": target,
                            "confidence": 0.8  # Default confidence
                        })
        except Exception as e:
            logger.error(f"Error parsing relationships: {e}")
        
        return relationships
    
    def _create_relationships_in_neo4j(self, relationships):
        """Create relationships in Neo4j"""
        with self.driver.session(database=self.database) as session:
            for rel in relationships:
                try:
                    session.run(f"""
                    MATCH (source) WHERE source.name = $source_name
                    MATCH (target) WHERE target.name = $target_name
                    MERGE (source)-[r:{rel['relationship']}]->(target)
                    SET r.confidence = $confidence,
                        r.source = 'llm_inference'
                    """, {
                        "source_name": rel["source"],
                        "target_name": rel["target"],
                        "confidence": rel["confidence"]
                    })
                except Exception as e:
                    logger.warning(f"Error creating relationship {rel}: {e}")
    
    def llm_analyze_document(self, doc_data):
        """Use LLM to analyze entire document for high-level insights"""
        try:
            title = doc_data.get("metadata", {}).get("title", "")
            content = ""
            
            # Combine content chunks
            for item in doc_data.get("content", []):
                if "text" in item:
                    content += item["text"] + "\n"
            
            # Limit content length for LLM
            content = content[:4000]
            
            chain = self.summarization_prompt | self.llm | StrOutputParser()
            analysis = chain.invoke({"title": title, "content": content})
            
            return analysis
        except Exception as e:
            logger.error(f"Error in document analysis: {e}")
            return ""
    
    def create_llm_enhanced_chunks(self, doc_id, content):
        """Create chunks with LLM-enhanced analysis"""
        try:
            chunks_created = 0
            if not content:
                logger.warning(f"No content found for document {doc_id}")
                return
                
            with self.driver.session(database=self.database) as session:
                for i, item in enumerate(tqdm(content, desc=f"Processing chunks for {doc_id}")):
                    if "text" in item:
                        chunk_id = f"{doc_id}_chunk_{i}"
                        chunk_text = item["text"]
                        chunk_type = item["type"]
                        
                        # Generate embedding
                        embedding = self.generate_embedding(chunk_text)
                        
                        # Create chunk node
                        session.run("""
                        MERGE (c:Chunk {id: $id})
                        SET c.text = $text,
                            c.type = $type,
                            c.embedding = $embedding
                        """, {
                            "id": chunk_id,
                            "text": chunk_text,
                            "type": chunk_type,
                            "embedding": embedding
                        })
                        
                        # Connect chunk to document
                        session.run("""
                        MATCH (d:Document {id: $doc_id})
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:PART_OF]->(d)
                        """, {
                            "doc_id": doc_id,
                            "chunk_id": chunk_id
                        })
                        
                        # Connect to previous chunk
                        if i > 0:
                            prev_chunk_id = f"{doc_id}_chunk_{i-1}"
                            session.run("""
                            MATCH (prev:Chunk {id: $prev_id})
                            MATCH (curr:Chunk {id: $curr_id})
                            MERGE (prev)-[:NEXT]->(curr)
                            """, {
                                "prev_id": prev_chunk_id,
                                "curr_id": chunk_id
                            })
                        
                        # LLM-enhanced entity extraction
                        entities = self.llm_extract_entities(chunk_text, chunk_id)
                        
                        # LLM-enhanced relationship inference
                        self.llm_infer_relationships(chunk_id, chunk_text)
                        
                        chunks_created += 1
            
            logger.info(f"Created {chunks_created} enhanced chunks for document {doc_id}")
        except Exception as e:
            logger.error(f"Error creating enhanced chunks for document {doc_id}: {e}")
    
    def llm_create_cross_document_relationships(self):
        """Use LLM to create relationships between chunks from different documents"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get pairs of chunks that might be related
                result = session.run("""
                MATCH (c1:Chunk)-[:PART_OF]->(d1:Document)
                MATCH (c2:Chunk)-[:PART_OF]->(d2:Document)
                WHERE id(c1) < id(c2) AND d1 <> d2
                WITH c1, c2, d1, d2
                ORDER BY rand()
                LIMIT 100
                RETURN c1.id as chunk1_id, c1.text as chunk1_text,
                       c2.id as chunk2_id, c2.text as chunk2_text
                """)
                
                relationships_created = 0
                for record in tqdm(result, desc="Analyzing cross-document relationships"):
                    chunk1_text = record["chunk1_text"][:1000]  # Limit text length
                    chunk2_text = record["chunk2_text"][:1000]
                    
                    # Use LLM to analyze relationship
                    chain = self.chunk_relationship_prompt | self.llm | StrOutputParser()
                    relationship_analysis = chain.invoke({
                        "chunk1": chunk1_text,
                        "chunk2": chunk2_text
                    })
                    
                    # Parse response and create relationship if found
                    if self._should_create_relationship(relationship_analysis):
                        rel_type, confidence = self._extract_relationship_info(relationship_analysis)
                        
                        session.run(f"""
                        MATCH (c1:Chunk {{id: $chunk1_id}})
                        MATCH (c2:Chunk {{id: $chunk2_id}})
                        MERGE (c1)-[r:{rel_type}]->(c2)
                        SET r.confidence = $confidence,
                            r.source = 'llm_inference'
                        """, {
                            "chunk1_id": record["chunk1_id"],
                            "chunk2_id": record["chunk2_id"],
                            "confidence": confidence
                        })
                        
                        relationships_created += 1
                
                logger.info(f"Created {relationships_created} cross-document relationships")
        except Exception as e:
            logger.error(f"Error creating cross-document relationships: {e}")
    
    def _should_create_relationship(self, analysis):
        """Determine if a relationship should be created based on LLM analysis"""
        return "NONE" not in analysis and any(
            rel_type in analysis for rel_type in 
            ["FOLLOWS", "REFERENCES", "SIMILAR", "PREREQUISITE", "EXAMPLE_OF"]
        )
    
    def _extract_relationship_info(self, analysis):
        """Extract relationship type and confidence from LLM analysis"""
        rel_type = "RELATED_TO"  # Default
        confidence = 0.7  # Default
        
        # Extract relationship type
        for rel in ["FOLLOWS", "REFERENCES", "SIMILAR", "PREREQUISITE", "EXAMPLE_OF"]:
            if rel in analysis:
                rel_type = rel
                break
        
        # Extract confidence
        import re
        conf_match = re.search(r'confidence[:\s]+([0-9.]+)', analysis)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        return rel_type, confidence
    
    def build_llm_enhanced_graph(self, docs_dir, doc_structure_path):
        """Main method to build LLM-enhanced knowledge graph"""
        logger.info("Starting LLM-enhanced graph building")
        
        # Create constraints
        self.create_constraints()
        
        try:
            # Load document structure
            with open(doc_structure_path, 'r', encoding='utf-8') as f:
                doc_structure = json.load(f)
            
            doc_ids = list(doc_structure.keys())
            logger.info(f"Found {len(doc_ids)} documents to process")
            
            # Process each document with LLM enhancement
            for doc_id in tqdm(doc_ids, desc="Building LLM-enhanced knowledge graph"):
                doc_path = os.path.join(docs_dir, f"{doc_id}.json")
                
                if os.path.exists(doc_path):
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                        
                        # Create document node
                        self.create_document_node(doc_data, doc_id)
                        
                        # LLM analysis of entire document
                        doc_analysis = self.llm_analyze_document(doc_data)
                        
                        # Store document analysis
                        with self.driver.session(database=self.database) as session:
                            session.run("""
                            MATCH (d:Document {id: $doc_id})
                            SET d.llm_analysis = $analysis
                            """, {"doc_id": doc_id, "analysis": doc_analysis})
                        
                        # Create LLM-enhanced chunks
                        if "content" in doc_data:
                            self.create_llm_enhanced_chunks(doc_id, doc_data["content"])
                        
                        # Create code examples
                        if "code_examples" in doc_data:
                            self.create_code_examples(doc_id, doc_data["code_examples"])
                        
                    except Exception as e:
                        logger.error(f"Error processing document {doc_id}: {e}")
                else:
                    logger.warning(f"Document file not found: {doc_path}")
            
            # Create document hierarchy
            self.create_document_hierarchy(doc_structure)
            
            # Create similarity links
            self.create_similarity_links()
            
            # Create cross-document relationships using LLM
            self.llm_create_cross_document_relationships()
            
            logger.info("LLM-enhanced graph building completed successfully!")
            
        except Exception as e:
            logger.error(f"Error building LLM-enhanced graph: {e}")
    
    def create_document_node(self, doc_data, doc_id):
        """Create document node"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                MERGE (d:Document {id: $id})
                SET d.title = $title,
                    d.url = $url,
                    d.description = $description
                """, {
                    "id": doc_id,
                    "title": doc_data["metadata"]["title"],
                    "url": doc_data["metadata"]["url"],
                    "description": doc_data["metadata"]["description"]
                })
        except Exception as e:
            logger.error(f"Error creating document node for {doc_id}: {e}")
    
    def create_document_hierarchy(self, doc_structure):
        """Create document hierarchy"""
        try:
            count = 0
            with self.driver.session(database=self.database) as session:
                for doc_id, doc_info in doc_structure.items():
                    if doc_info.get("parent"):
                        parent_id = doc_info["parent"]
                        session.run("""
                        MATCH (parent:Document {id: $parent_id})
                        MATCH (child:Document {id: $child_id})
                        MERGE (parent)-[:CONTAINS]->(child)
                        """, {
                            "parent_id": parent_id,
                            "child_id": doc_id
                        })
                        count += 1
            logger.info(f"Created {count} hierarchical relationships")
        except Exception as e:
            logger.error(f"Error creating document hierarchy: {e}")
    
    def create_code_examples(self, doc_id, code_examples):
        """Create code examples"""
        try:
            if not code_examples:
                return
                
            with self.driver.session(database=self.database) as session:
                for i, code in enumerate(code_examples):
                    example_id = f"{doc_id}_example_{i}"
                    
                    session.run("""
                    MERGE (e:CodeExample {id: $id})
                    SET e.code = $code
                    """, {
                        "id": example_id,
                        "code": code
                    })
                    
                    session.run("""
                    MATCH (d:Document {id: $doc_id})
                    MATCH (e:CodeExample {id: $example_id})
                    MERGE (e)-[:EXAMPLE_OF]->(d)
                    """, {
                        "doc_id": doc_id,
                        "example_id": example_id
                    })
            
            logger.info(f"Created {len(code_examples)} code examples for document {doc_id}")
        except Exception as e:
            logger.error(f"Error creating code examples for document {doc_id}: {e}")
    
    def create_similarity_links(self):
        """Create similarity links"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                MATCH (c1:Chunk)
                CALL db.index.vector.queryNodes('chunk_embeddings', 5, c1.embedding) 
                YIELD node AS c2, score
                WHERE id(c1) <> id(c2) AND score > 0.8
                MERGE (c1)-[:SIMILAR {score: score}]->(c2)
                """)
            logger.info("Created similarity links between chunks")
        except Exception as e:
            logger.error(f"Error creating similarity links: {e}")


# Additional utility functions and classes
class AdvancedLLMGraphOperations:
    """Advanced operations for LLM-enhanced knowledge graphs"""
    
    def __init__(self, graph_builder: LLMEnhancedGraphBuilder):
        self.builder = graph_builder
        self.llm = graph_builder.llm
        self._setup_advanced_prompts()
    
    def _setup_advanced_prompts(self):
        """Setup advanced prompts for specialized graph operations"""
        
        # Prompt for identifying missing relationships
        self.missing_relationships_prompt = ChatPromptTemplate.from_template("""
        You are analyzing a knowledge graph about Shiny for Python documentation.
        
        Current entities and their existing relationships:
        {current_state}
        
        Based on your knowledge of Shiny for Python, what important relationships 
        are likely missing from this graph? Consider:
        
        1. **Dependency relationships**: What components depend on what functions?
        2. **Workflow relationships**: What is the typical order of using these entities?
        3. **Conceptual groupings**: What entities belong to the same conceptual area?
        4. **Best practice relationships**: What entities work well together?
        
        Suggest up to 10 missing relationships with:
        - source_entity: Name of source entity
        - relationship_type: Type of relationship (DEPENDS_ON, WORKS_WITH, PART_OF, ENABLES, etc.)
        - target_entity: Name of target entity
        - confidence: Your confidence (0.0-1.0)
        - reasoning: Why this relationship should exist
        
        Only suggest relationships with confidence > 0.7.
        """)
        
        # Prompt for semantic clustering
        self.clustering_prompt = ChatPromptTemplate.from_template("""
        You are analyzing entities in a Shiny knowledge graph to identify semantic clusters.
        
        Entities to cluster:
        {entities}
        
        Group these entities into meaningful clusters based on:
        - Functional similarity (what they do)
        - Usage patterns (when they're used together)
        - Conceptual relationships (what concepts they represent)
        - User workflow (how users typically encounter them)
        
        For each cluster, provide:
        - cluster_name: Descriptive name for the cluster
        - entities: List of entities in this cluster
        - description: What this cluster represents
        - typical_use_cases: When users would work with this cluster
        
        Aim for 5-8 meaningful clusters.
        """)
    
    def identify_missing_relationships(self):
        """Use LLM to identify potentially missing relationships in the graph"""
        try:
            with self.builder.driver.session(database=self.builder.database) as session:
                # Get current graph state
                result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN 
                    labels(n)[0] as source_type,
                    n.name as source_name,
                    type(r) as relationship,
                    labels(m)[0] as target_type,
                    m.name as target_name
                LIMIT 50
                """)
                
                current_relationships = []
                for record in result:
                    current_relationships.append({
                        "source_type": record["source_type"],
                        "source_name": record["source_name"],
                        "relationship": record["relationship"],
                        "target_type": record["target_type"],
                        "target_name": record["target_name"]
                    })
                
                # Get entity counts
                stats = session.run("""
                RETURN 
                    size((n:Component)) as components,
                    size((n:Function)) as functions,
                    size((n:Topic)) as topics
                """).single()
                
                current_state = {
                    "relationships": current_relationships,
                    "stats": dict(stats)
                }
                
                # Use LLM to identify missing relationships
                chain = self.missing_relationships_prompt | self.llm | StrOutputParser()
                suggestions = chain.invoke({"current_state": json.dumps(current_state, indent=2)})
                
                # Parse and potentially create suggested relationships
                missing_rels = self._parse_missing_relationships(suggestions)
                self._create_suggested_relationships(missing_rels)
                
                logger.info(f"Identified and created {len(missing_rels)} missing relationships")
                return missing_rels
                
        except Exception as e:
            logger.error(f"Error identifying missing relationships: {e}")
            return []
    
    def _parse_missing_relationships(self, suggestions):
        """Parse LLM suggestions for missing relationships"""
        relationships = []
        try:
            # Simple parsing - look for structured suggestions
            lines = suggestions.split('\n')
            current_rel = {}
            
            for line in lines:
                line = line.strip()
                if 'source_entity:' in line:
                    current_rel['source'] = line.split(':', 1)[1].strip()
                elif 'relationship_type:' in line:
                    current_rel['type'] = line.split(':', 1)[1].strip()
                elif 'target_entity:' in line:
                    current_rel['target'] = line.split(':', 1)[1].strip()
                elif 'confidence:' in line:
                    try:
                        current_rel['confidence'] = float(line.split(':', 1)[1].strip())
                    except:
                        current_rel['confidence'] = 0.7
                elif 'reasoning:' in line:
                    current_rel['reasoning'] = line.split(':', 1)[1].strip()
                    # Complete relationship entry
                    if all(k in current_rel for k in ['source', 'type', 'target']):
                        relationships.append(current_rel.copy())
                        current_rel = {}
        except Exception as e:
            logger.error(f"Error parsing missing relationships: {e}")
        
        return relationships
    
    def _create_suggested_relationships(self, relationships):
        """Create suggested relationships in the graph"""
        with self.builder.driver.session(database=self.builder.database) as session:
            created_count = 0
            for rel in relationships:
                try:
                    if rel.get('confidence', 0) > 0.7:
                        session.run(f"""
                        MATCH (source) WHERE source.name = $source_name
                        MATCH (target) WHERE target.name = $target_name
                        MERGE (source)-[r:{rel['type']}]->(target)
                        SET r.confidence = $confidence,
                            r.reasoning = $reasoning,
                            r.source = 'llm_inference'
                        """, {
                            "source_name": rel['source'],
                            "target_name": rel['target'],
                            "confidence": rel.get('confidence', 0.8),
                            "reasoning": rel.get('reasoning', 'LLM inferred relationship')
                        })
                        created_count += 1
                except Exception as e:
                    logger.warning(f"Could not create relationship {rel}: {e}")
            
            logger.info(f"Successfully created {created_count} suggested relationships")
    
    def create_semantic_clusters(self):
        """Use LLM to create semantic clusters of related entities"""
        try:
            with self.builder.driver.session(database=self.builder.database) as session:
                # Get all entities
                entities_result = session.run("""
                MATCH (n)
                WHERE n:Component OR n:Function OR n:Topic
                RETURN labels(n)[0] as type, n.name as name, n.description as description
                """)
                
                entities = [dict(record) for record in entities_result]
                
                # Use LLM for clustering
                chain = self.clustering_prompt | self.llm | StrOutputParser()
                clustering_result = chain.invoke({
                    "entities": json.dumps(entities, indent=2)
                })
                
                # Parse clustering results and create cluster nodes
                clusters = self._parse_clustering_results(clustering_result)
                self._create_cluster_nodes(clusters)
                
                logger.info(f"Created {len(clusters)} semantic clusters")
                return clusters
                
        except Exception as e:
            logger.error(f"Error creating semantic clusters: {e}")
            return []
    
    def _parse_clustering_results(self, clustering_result):
        """Parse LLM clustering results"""
        clusters = []
        try:
            # Simple parsing for cluster information
            lines = clustering_result.split('\n')
            current_cluster = {}
            
            for line in lines:
                line = line.strip()
                if 'cluster_name:' in line:
                    current_cluster = {'name': line.split(':', 1)[1].strip()}
                elif 'entities:' in line:
                    entities_str = line.split(':', 1)[1].strip()
                    # Parse entity list (assuming comma-separated)
                    current_cluster['entities'] = [e.strip() for e in entities_str.split(',')]
                elif 'description:' in line:
                    current_cluster['description'] = line.split(':', 1)[1].strip()
                elif 'typical_use_cases:' in line:
                    current_cluster['use_cases'] = line.split(':', 1)[1].strip()
                    # Complete cluster entry
                    if 'name' in current_cluster and 'entities' in current_cluster:
                        clusters.append(current_cluster.copy())
                        current_cluster = {}
        except Exception as e:
            logger.error(f"Error parsing clustering results: {e}")
        
        return clusters
    
    def _create_cluster_nodes(self, clusters):
        """Create cluster nodes in the graph"""
        with self.builder.driver.session(database=self.builder.database) as session:
            for cluster in clusters:
                try:
                    # Create cluster node
                    cluster_id = f"cluster_{cluster['name'].lower().replace(' ', '_')}"
                    session.run("""
                    MERGE (c:Cluster {id: $id})
                    SET c.name = $name,
                        c.description = $description,
                        c.use_cases = $use_cases
                    """, {
                        "id": cluster_id,
                        "name": cluster['name'],
                        "description": cluster.get('description', ''),
                        "use_cases": cluster.get('use_cases', '')
                    })
                    
                    # Link entities to cluster
                    for entity_name in cluster.get('entities', []):
                        session.run("""
                        MATCH (c:Cluster {id: $cluster_id})
                        MATCH (e) WHERE e.name = $entity_name
                        MERGE (e)-[:BELONGS_TO]->(c)
                        """, {
                            "cluster_id": cluster_id,
                            "entity_name": entity_name.strip()
                        })
                        
                except Exception as e:
                    logger.warning(f"Error creating cluster {cluster.get('name', 'unknown')}: {e}")
    
    def optimize_graph_structure(self):
        """Perform comprehensive LLM-powered graph optimization"""
        logger.info("🔧 Starting comprehensive graph optimization...")
        
        try:
            # Step 1: Identify missing relationships
            logger.info("🔍 Identifying missing relationships...")
            missing_rels = self.identify_missing_relationships()
            
            # Step 2: Create semantic clusters
            logger.info("🏷️  Creating semantic clusters...")
            clusters = self.create_semantic_clusters()
            
            # Step 3: Create summary report
            optimization_report = {
                "missing_relationships_added": len(missing_rels),
                "semantic_clusters_created": len(clusters),
                "optimization_timestamp": str(datetime.datetime.now())
            }
            
            # Store optimization report in the graph
            with self.builder.driver.session(database=self.builder.database) as session:
                session.run("""
                MERGE (r:OptimizationReport {id: 'latest'})
                SET r.report = $report,
                    r.timestamp = datetime()
                """, {"report": json.dumps(optimization_report, indent=2)})
            
            logger.info("✅ Graph optimization completed successfully!")
            return optimization_report
            
        except Exception as e:
            logger.error(f"❌ Error during graph optimization: {e}")
            return {"error": str(e)}


# Integration function for complete pipeline
def build_and_optimize_knowledge_graph(docs_dir, doc_structure_path):
    """Complete pipeline for building and optimizing an LLM-enhanced knowledge graph"""
    
    # Initialize the graph builder
    builder = LLMEnhancedGraphBuilder(
        uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )
    
    try:
        print("🚀 Phase 1: Building LLM-Enhanced Knowledge Graph...")
        builder.build_llm_enhanced_graph(docs_dir, doc_structure_path)
        
        print("🔧 Phase 2: Advanced Graph Optimization...")
        optimizer = AdvancedLLMGraphOperations(builder)
        optimization_report = optimizer.optimize_graph_structure()
        
        print("📋 Optimization Report:")
        print(json.dumps(optimization_report, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    finally:
        builder.close()


# Usage example
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example usage
    docs_directory = "shiny_docs\content"
    doc_structure_file = "doc_structure.json"
    
    success = build_and_optimize_knowledge_graph(docs_directory, doc_structure_file)
    
    if success:
        print("✅ Knowledge graph built and optimized successfully!")
    else:
        print("❌ Failed to build knowledge graph")