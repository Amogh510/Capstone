# Knowledge Graph Retrieval Service

A FastAPI-based retrieval system for extracting relevant context from UI codebase knowledge graphs stored in Neo4j. Designed specifically for LLM integration to maintain context over large codebases.

## 🚀 Features

- **Multiple Search Types**: Fulltext, component-centric, file-based, and semantic search
- **Context Expansion**: Automatically expand search results by following relationships
- **LLM-Optimized Output**: Pre-formatted context strings ready for LLM consumption  
- **Smart Filtering**: Filter by file paths, components, or other criteria
- **RESTful API**: Easy integration with any system
- **Python Client**: Ready-to-use client library for seamless integration

## 🏗 Architecture

```
User Query → Retrieval Service → Neo4j KG → Context Expansion → LLM-Formatted Output
```

## 📦 Installation

1. **Install dependencies:**
   ```bash
   cd retrieval-service
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Neo4j credentials
   ```

3. **Start the service:**
   ```bash
   python main.py
   ```

The service will be available at `http://localhost:8000`

## 🔍 Search Types

### 1. **Fulltext Search**
```python
client.search("Alert component", search_type="fulltext")
```
Searches node names, file paths, and component names.

### 2. **Component Search** 
```python
client.search_component("Alert", expand_depth=2)
```
Finds components and their relationships (props, hooks, JSX elements, etc.).

### 3. **File Search**
```python
client.search_file("components/Alert/Alert.tsx")
```
Gets all nodes from a specific file.

### 4. **Semantic Search** (Future)
Enhanced search using embeddings (currently falls back to fulltext).

## 🛠 API Endpoints

### Main Search
```http
POST /search
{
  "query": "Alert component",
  "search_type": "component",
  "max_results": 10,
  "expand_depth": 1,
  "file_filter": "Alert.tsx"
}
```

### Quick Searches
```http
GET /search/component/{component_name}?expand_depth=1&max_results=20
GET /search/file?file_path=Alert.tsx&max_results=50  
GET /search/fulltext?q=Alert&max_results=10&expand_depth=0
```

### Utilities
```http
GET /health          # Health check
GET /stats           # Database statistics
```

## 🐍 Python Client Usage

### Basic Usage
```python
from client import RetrievalClient

client = RetrievalClient()

# Get LLM-ready context
context = client.get_llm_context(
    query="How does the Alert component work?",
    search_type="component",
    expand_depth=2
)

# Use with your LLM
llm_response = your_llm.complete(f"{context}\n\nUser: {user_question}")
```

### Advanced Usage with Context Builder
```python
from client import RetrievalClient, LLMContextBuilder

client = RetrievalClient()
builder = LLMContextBuilder(client)

# Smart context building for questions
context = builder.build_context_for_question(
    "How do I modify the Alert component to support custom icons?"
)

# File-specific context
context = builder.build_context_for_file_question(
    "src/components/Alert/Alert.tsx",
    "What props does this component accept?"
)

# Component-specific context
context = builder.build_context_for_component_question(
    "Alert",
    "How is this component styled?",
    include_related=True
)
```

## 📊 Context Output Format

The service provides LLM-optimized context in this format:

```markdown
# Codebase Context for Query: 'Alert component'

## Summary
- Found 15 nodes across 3 files
- 2 components involved  
- 12 relationships

## File: src/components/Alert/Alert.tsx
### Components:
- **Alert** (default export)
  - Props: ['title', 'severity', 'onClose']
  - Hooks: ['useState', 'useCallback']

### JSX Elements:
- div: 3 instances
- button: 1 instances  
- span: 2 instances

## Key Relationships:
- componentRendersJsx: 6 connections
- componentHasProp: 3 connections
- componentUsesHook: 2 connections
```

## 🎯 LLM Integration Examples

### With OpenAI
```python
import openai
from client import LLMContextBuilder, RetrievalClient

client = RetrievalClient()
builder = LLMContextBuilder(client)

def ask_codebase_question(question: str) -> str:
    # Get relevant context
    context = builder.build_context_for_question(question)
    
    # Build prompt
    prompt = f"""You are a helpful assistant with access to a UI codebase knowledge graph.

{context}

User Question: {question}

Please answer based on the provided codebase context."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    return response.choices[0].message.content
```

### With Local LLM (Ollama)
```python
import requests
from client import LLMContextBuilder, RetrievalClient

def ask_with_ollama(question: str, model: str = "codellama"):
    client = RetrievalClient()
    builder = LLMContextBuilder(client)
    
    context = builder.build_context_for_question(question)
    
    prompt = f"Codebase Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    
    return response.json()["response"]
```

## 🔧 Configuration

Environment variables in `.env`:

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j  
NEO4J_PASSWORD=admin123

# Service
MAX_CONTEXT_LENGTH=8000
API_HOST=0.0.0.0
API_PORT=8000
```

## 🚀 Performance

- **Response Time**: < 100ms for most queries
- **Context Size**: Configurable, default 8000 chars
- **Concurrent Requests**: Supports multiple concurrent searches
- **Memory Usage**: ~100MB base + Neo4j connection pool

## 🛣 Roadmap

- [ ] **Semantic Search**: Add embedding-based search using sentence-transformers
- [ ] **Caching**: Redis cache for frequent queries  
- [ ] **Code Snippets**: Include actual source code in context
- [ ] **Relevance Scoring**: Improve result ranking
- [ ] **GraphQL API**: Alternative to REST
- [ ] **WebSocket**: Real-time search updates

## 📝 Example Queries

Try these in your LLM with the retrieved context:

1. **"How does the Alert component handle different severity levels?"**
2. **"What hooks are used in the user authentication flow?"**
3. **"Which components use the useLocalStorage hook?"**  
4. **"How is routing implemented in this application?"**
5. **"What CSS classes are applied to form inputs?"**

## 🐛 Troubleshooting

1. **Service won't start**: Check Neo4j connection and credentials
2. **Empty results**: Verify KG data was imported correctly  
3. **Slow responses**: Reduce `expand_depth` or `max_results`
4. **Memory issues**: Lower `MAX_CONTEXT_LENGTH`
