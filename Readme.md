# LLM-Enhanced Knowledge Graph System (GraphRAG)

## 🧠 **What is GraphRAG?**

**GraphRAG** = **Graph** + **RAG** (Retrieval-Augmented Generation)

Traditional RAG: `User Query → Vector Search → Retrieved Chunks → LLM → Answer`

GraphRAG: `User Query → Graph Traversal + Vector Search → Related Entities + Relationships + Context → LLM → Enhanced Answer`

This system transforms your documentation into an intelligent knowledge graph that understands relationships, concepts, and workflows - then uses that understanding to provide contextual, comprehensive answers.

Original Microsoft Blog: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
What is Cypher? : https://neo4j.com/docs/getting-started/cypher/

## 🏗️ **System Architecture**

### **Phase 1: Graph Construction**
```
Raw Documents → LLM Analysis → Knowledge Graph → Enhanced Graph
```

### **Phase 2: Query Processing (GraphRAG)**
```
Natural Language Query → LLM-Generated Cypher → Graph Traversal → Contextual Results → Enhanced Answer
```

## 🔬 **How It Works**

### **1. Document Processing & Chunking**
```python
# Takes your JSON documents and creates intelligent chunks
for document in documents:
    chunks = create_chunks(document)
    embeddings = generate_embeddings(chunks)
    store_in_neo4j(chunks, embeddings)
```

**What happens:**
- Splits documents into manageable chunks
- Creates vector embeddings for semantic similarity
- Stores both text and embeddings in Neo4j graph database

### **2. LLM-Enhanced Entity Extraction**
```python
# LLM analyzes each chunk to extract meaningful entities
entities = llm.extract({
    "components": "UI elements (ui.input_text, ui.output_plot)",
    "functions": "Python functions (@render.text, server functions)",
    "topics": "Key concepts (reactivity, layouts, modules)"
})
```

**Why this matters:** Instead of regex patterns, the LLM **understands context** and extracts entities with semantic meaning.

### **3. Relationship Inference**
```python
# LLM analyzes relationships between entities
relationships = llm.infer_relationships({
    "components": ["ui.input_text", "ui.output_plot"],
    "functions": ["@render.text", "@reactive.calc"],
    "context": chunk_text
})
```

**Creates relationships like:**
```cypher
(ui.input_text)-[:USES]->(render.text)
(reactivity)-[:RELATES_TO]->(ui.input_text)
(dashboard)-[:DEPENDS_ON]->(layout)
```

### **4. Semantic Clustering**
```python
# LLM groups related entities into semantic clusters
clusters = llm.create_clusters(entities)
```

**Example clusters:**
- **Input Cluster**: `ui.input_text, ui.input_slider, ui.input_select`
- **Layout Cluster**: `ui.layout_columns, ui.sidebar, ui.nav_panel`
- **Reactivity Cluster**: `@render.text, @reactive.calc, reactive.Value`

### **5. Intelligent Query Processing**

**Traditional RAG:**
```
Query: "How do I create interactive inputs?"
↓
Vector search in chunks → Random chunks mentioning "input"
↓
LLM generates answer from disconnected chunks
```

**Our GraphRAG:**
```
Query: "How do I create interactive inputs?"
↓
1. LLM generates intelligent Cypher query
2. Graph traversal finds:
   - Input components (ui.input_text, ui.input_slider)
   - Related functions (@render.text, @reactive.calc)
   - Code examples showing usage
   - Semantic cluster "Input Components"
   - Dependencies and workflows
↓
Rich contextual answer with relationships
```

## 🧩 **Why This is True GraphRAG**

### **1. Knowledge Graph Foundation**
- **Entities**: Components, Functions, Topics extracted by LLM
- **Relationships**: Semantic relationships inferred by LLM with confidence scores
- **Structure**: Hierarchical document organization

### **2. Graph-Native Retrieval**
```cypher
// Can find complex patterns like:
MATCH (input:Component)-[:USES]->(render:Function)
WHERE input.name CONTAINS "input"
MATCH (render)-[:ENABLES]->(output:Component)
RETURN input, render, output
```

### **3. Multi-Level Fallback Strategy**
```python
# If primary query fails, try:
1. Semantic cluster search
2. LLM-inferred relationship search
3. Enhanced text search with document analysis
4. Generic fallback with confidence scoring
```

### **4. Enhanced Response Generation**
- Direct answer to query
- Related concepts from graph traversal
- Code examples from connected nodes
- Workflow information from relationships
- Contextual follow-up questions

## 🔧 **The LLM's Multiple Roles**

1. **Graph Builder**: Extracts entities and relationships from text
2. **Query Planner**: Converts natural language to intelligent Cypher queries
3. **Context Enhancer**: Analyzes document semantics and dependencies
4. **Response Generator**: Creates comprehensive, contextual answers
5. **Relationship Discoverer**: Finds missing connections and semantic clusters

## 🚀 **Key Advantages Over Traditional RAG**

### **Traditional RAG Limitations:**
- ❌ Retrieves isolated chunks without context
- ❌ No understanding of relationships between concepts
- ❌ Limited ability to answer "how do X and Y work together?"
- ❌ Can't provide workflow or dependency information

### **Our GraphRAG Solutions:**
- ✅ **Relationship-aware**: Understands how concepts connect
- ✅ **Context-rich**: Provides semantic context and workflows
- ✅ **Comprehensive**: Finds related concepts you didn't explicitly ask for
- ✅ **Intelligent**: LLM understands intent and generates smart queries
- ✅ **Confidence-scored**: Each relationship has a confidence level
- ✅ **Fallback strategies**: Multiple ways to find relevant information

## 🔬 **Concrete Example**

**User asks**: *"How do dashboard layouts work with input components?"*

**GraphRAG Process:**

1. **Query Analysis**: LLM generates Cypher to find dashboard-input relationships

2. **Graph Traversal**:
```cypher
MATCH (dashboard:Topic)-[:RELATES_TO]->(layout:Component)
MATCH (layout)-[:WORKS_WITH]->(input:Component)
MATCH (input)-[:MENTIONED_IN]->(chunk:Chunk)
RETURN dashboard, layout, input, chunk.text
```

3. **Retrieved Context**:
   - Dashboard layout components and their properties
   - How layouts connect to input components
   - Code examples showing integration patterns
   - Best practices from LLM analysis
   - Related semantic clusters

4. **Enhanced Answer**:
   - Explains dashboard-input relationships with context
   - Shows relevant code examples
   - Suggests related components and workflows
   - Provides follow-up questions based on graph connections

## 📋 **Prerequisites**

- Python 3.8+
- Neo4j 4.0+ (with APOC plugin recommended)
- OpenAI API key
- Minimum 8GB RAM (16GB recommended for large documents)

## 🚀 **Quick Start**

### **1. Clone and Setup**
```bash
# Clone your repository (or create these files)
git clone <your-repo>
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

### **2. Environment Setup**
Create a `.env` file:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Optional: Customize LLM model
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0
```

### **3. Prepare Your Data**
Ensure you have the correct directory structure:
```
your_project/
├── .env
├── llm_graph_builder.py
├── enhanced_query_system.py
├── quick_start.py
├── requirements.txt
└── shiny_docs/
    ├── doc_structure.json
    └── content/
        ├── document1.json
        ├── document2.json
        └── ...
```

### **4. Check Setup**
```bash
python quick_start.py check
```

### **5. Build Knowledge Graph**
```bash
python quick_start.py build
```

### **6. Run Interactive Queries**
```bash
python quick_start.py query
```

### **7. Validate Graph Features**
```bash
python quick_start.py validate
```

## 🛠️ **Alternative Usage Methods**

### **Method 1: Using quick_start.py (Recommended)**
```bash
# Check your setup
python quick_start.py check

# Build the enhanced knowledge graph
python quick_start.py build

# Run interactive queries
python quick_start.py query


```

## 📊 **Understanding the Process**

### **Building Phase (Can take several hours)**
```
🚀 Phase 1: Building LLM-Enhanced Knowledge Graph...
├── 📄 Processing 473 documents
├── 🧠 LLM analyzing each chunk for entities
├── 🔗 Inferring relationships between entities
├── 📊 Creating document hierarchy
├── 🎯 Generating similarity links
└── 🌐 Creating cross-document relationships

🔧 Phase 2: Advanced Graph Optimization...
├── 🔍 Identifying missing relationships
├── 🏷️  Creating semantic clusters
└── 📋 Generating optimization report
```

### **Query Phase (Real-time)**
```
🔍 User Query → 🧠 LLM Analysis → 📊 Graph Traversal → 📝 Enhanced Response
```

## 💡 **Example Queries to Try**

Once your graph is built, try these example queries:

### **Basic Queries**
- "How do I create interactive inputs?"
- "Show me examples of reactivity"
- "What components work well together?"
- "How do I build a dashboard layout?"

### **Relationship Queries**
- "What functions work with input components?"
- "How does reactivity connect to UI components?"
- "What are the dependencies for building dashboards?"

### **Workflow Queries**
- "What's the typical workflow for building a Shiny app?"
- "How do I integrate multiple input types?"
- "What components are commonly used together?"

## 🔧 **Configuration Options**

### **LLM Configuration**
In `enhanced_query_system.py`:
```python
self.llm = ChatOpenAI(
    model="gpt-4o",           # or "gpt-3.5-turbo" for cost efficiency
    temperature=0,            # 0 for deterministic, 0.7 for creative
    max_tokens=2000,          # Adjust based on needs
    timeout=30                # Request timeout
)
```

### **Graph Building Configuration**
In `llm_graph_builder.py`:
```python
self.max_chunk_length = 1000        # Max characters per chunk
self.confidence_threshold = 0.7     # Minimum confidence for relationships
self.max_entities_per_chunk = 10    # Limit entities per chunk
self.enable_cross_doc_relations = True  # Enable cross-document relationships
```

## 📊 **Monitoring and Performance**

### **Get System Statistics**
```python
from enhanced_query_system import EnhancedShinyQuerySystem

system = EnhancedShinyQuerySystem()
stats = system.get_graph_statistics()
print(f"Total entities: {sum(stats.values())}")
print(f"LLM relationships: {stats['llm_relationships']}")
```

### **Validate Enhanced Features**
```python
from enhanced_query_system import validate_enhanced_graph
validation_result = validate_enhanced_graph()
```

## 🐛 **Troubleshooting**

### **Common Issues**

1. **No LLM Relationships Found**
   ```bash
   python quick_start.py validate
   ```

2. **Query Performance Issues**
   ```cypher
   // Check Neo4j indexes
   SHOW INDEXES
   ```

3. **Memory Issues**
   - Reduce `max_chunk_length` in configuration
   - Process documents in smaller batches
   - Increase system memory allocation

4. **API Rate Limits**
   - Add delays between LLM calls
   - Use GPT-3.5-turbo instead of GPT-4
   - Implement request batching

### **Debug Mode**
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 **Performance Optimization**

### **For Large Document Collections**
```python
# Batch processing
batch_size = 50
for i in range(0, len(doc_ids), batch_size):
    batch = doc_ids[i:i+batch_size]
    builder.process_document_batch(batch)

# Parallel processing
builder.enable_parallel_processing = True
builder.max_workers = 4
```

### **For High-Frequency Queries**
```python
# Enable query result caching
system.enable_query_cache = True
system.cache_duration = 3600  # 1 hour
```

## 🔌 **Integration Examples**

### **Web API Integration**
```python
from flask import Flask, request, jsonify
from enhanced_query_system import EnhancedShinyQuerySystem

app = Flask(__name__)
system = EnhancedShinyQuerySystem()

@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    query = data.get('query', '')
    
    result = system.query_enhanced_shiny_docs(query)
    
    return jsonify({
        'answer': result['final_answer'],
        'follow_up_questions': result['follow_up_questions'],
        'context': result.get('context_info', {})
    })
```

### **Streamlit Dashboard**
```python
import streamlit as st
from enhanced_query_system import EnhancedShinyQuerySystem

st.title("Enhanced Shiny Documentation Assistant")

@st.cache_resource
def load_system():
    return EnhancedShinyQuerySystem()

system = load_system()
query = st.text_input("Ask about Shiny for Python:")

if query:
    with st.spinner("Processing your query..."):
        result = system.query_enhanced_shiny_docs(query)
    
    st.write("### Answer")
    st.write(result['final_answer'])
    
    if result.get('follow_up_questions'):
        st.write("### Follow-up Questions")
        for q in result['follow_up_questions']:
            if st.button(q):
                st.rerun()
```

## 📂 **File Structure**

```
project/
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── llm_graph_builder.py         # Core graph building logic
├── enhanced_query_system.py     # Query processing system
├── build_graph.py              # Standalone graph builder
├── quick_start.py               # All-in-one management script
├── README.md                    # This file
└── shiny_docs/                  # Your documentation
    ├── doc_structure.json       # Document hierarchy
    └── content/                 # JSON document files
        ├── document1.json
        ├── document2.json
        └── ...
```

## 🎯 **Expected Results**

After building your graph, you should see:
- **Entities**: LLM-extracted components, functions, topics
- **Relationships**: Confidence-scored connections between entities
- **Clusters**: Semantic groupings of related concepts
- **Enhanced Queries**: Intelligent responses with context and examples

The system transforms your static documentation into an intelligent assistant that understands relationships, workflows, and can answer complex questions about how different concepts work together.

## 🤝 **Contributing**

This system is designed to be extensible. You can:
- Add new entity types in the LLM prompts
- Modify relationship inference logic
- Extend the query generation strategies
- Add new fallback mechanisms

## 📄 **License**

[Add your license information here]

---

**This LLM-Enhanced GraphRAG system transforms your documentation from a static collection of files into an intelligent, relationship-aware knowledge assistant that can answer complex questions about concepts, workflows, and dependencies.** 🚀🧠
