## Complete Test Generation Pipeline — Stages 1, 2A & 3

Implements the complete UI Test Case Generation Pipeline:
- **Stage 1A**: KG Entity Extraction (Routes + Components) via semantic similarity
- **Stage 1B**: LLM-based Structured Scenario Interpretation
- **Stage 2A**: Subgraph Retrieval (Context Extraction from KG)
- **Stage 3**: E2E Test Case Generation (Playwright/Cypress/Selenium)

### Features

#### Stage 1A: Multi-Stage Retrieval
- **Global query search**: Embeds entire scenario for holistic understanding
- **Per-step search**: Individual embeddings for each step with position-aware weighting
- **Keyword boosting**: Additional scoring for exact route matches
- Neo4j vector index search (preferred if indexes exist)
- Local cosine-similarity fallback with cached embeddings
- LangChain embeddings (HuggingFace) by default

#### Stage 1B: LLM Structured Interpretation
- Uses retrieved KG entities to produce structured JSON output
- Supports any LiteLLM-compatible model (GPT-4, Claude, Groq, etc.)
- Validates scenario type, required routes/components, auth requirements
- Determines priority based on workflow complexity

#### Stage 2A: Subgraph Retrieval (Smart Filtering)
- **🎯 Smart Filtering**: Removes noise while keeping test-relevant context
  - **Minimal mode** (default): 82 nodes, 94 edges (78% reduction)
  - Focuses on: Components, Routes, States, EventHandlers, Hooks, Props
  - Excludes: Styling nodes (TailwindUtility, InlineStyle), deep JSX trees
- **📊 Three filtering levels**:
  - **Minimal**: Test-relevant entities only (recommended for test generation)
  - **Smart Filtered**: Balanced view with aggregated styling
  - **Full**: Complete graph with all details
- **🎨 Styling aggregation**: Summarizes Tailwind classes instead of 1300+ individual nodes
- **🌳 JSX depth limiting**: Prevents overwhelming DOM tree details
- **Configurable depth traversal**: Control how deep to explore relationships (default: depth=2)
- **Efficient graph queries**: Optimized Cypher queries for fast subgraph extraction
- **Summary statistics**: Node/edge counts, type breakdowns, and styling summaries

#### Stage 3: E2E Test Case Generation
- **🧪 Multi-Framework Support**: Generate tests for Playwright, Cypress, or Selenium
- **🤖 LLM-Powered**: Uses advanced language models (GPT-4, Claude, Llama, etc.) via LiteLLM
- **🎯 Context-Aware**: Leverages KG subgraph to generate accurate selectors and flows
- **📝 Production-Ready**: Generates complete, executable tests with proper structure
- **✅ Comprehensive Assertions**: Includes navigation checks, state verification, and UI validation
- **🔐 Auth Support**: Automatically includes login flows when required
- **📚 Best Practices**: Follows framework-specific patterns and conventions
- **💡 Smart Selectors**: Prefers data-testid, role-based, and semantic selectors
- **⚡ Robust Waiting**: Includes proper waits and error handling

### Environment

#### Required
- `NEO4J_URI` (default `bolt://localhost:7687`)
- `NEO4J_USER` (default `neo4j`)
- `NEO4J_PASS` or `NEO4J_PASSWORD` (default `admin123`)

#### Stage 1A Configuration
- `STAGE1_EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `STAGE1_COMPONENT_INDEX` (default `component_embeddings`)
- `STAGE1_ROUTE_INDEX` (default `route_embeddings`)
- `STAGE1_TOPK_COMPONENTS` (default `5`)
- `STAGE1_TOPK_ROUTES` (default `5`)
- `STAGE1_CACHE_DIR` (default `.cache/stage1`)

#### Stage 1B Configuration (LLM)
- `LITELLM_MODEL` (e.g., `groq/llama-3.3-70b-versatile`, `gpt-4o-mini`)
- `GROQ_API_KEY` (if using Groq models)
- `OPENAI_API_KEY` (if using OpenAI models)
- `STAGE1_ENABLE_STAGE1B` (default `true`)
- `STAGE1_LLM_TEMPERATURE` (default `0.0`)
- `STAGE1_LLM_MAX_TOKENS` (default `500`)

#### Stage 2A Configuration (Subgraph Retrieval)
- `STAGE2_MAX_DEPTH` (default `2`)
- `STAGE2_INCLUDE_FILE_CONTEXT` (default `true`)

#### Stage 3 Configuration (Test Generation)
- `STAGE3_TEST_FRAMEWORK` (default `playwright`) - Options: `playwright`, `cypress`, `selenium`
- `STAGE3_LLM_MODEL` (e.g., `groq/llama-3.3-70b-versatile`, `gpt-4o-mini`)
- `STAGE3_TEMPERATURE` (default `0.1`) - LLM temperature for test generation
- `STAGE3_MAX_TOKENS` (default `2000`) - Max tokens for generated test
- `STAGE3_INCLUDE_COMMENTS` (default `true`) - Include inline comments
- `STAGE3_INCLUDE_ASSERTIONS` (default `true`) - Include assertions in tests

### Install (recommended: reuse retrieval-service venv or create a new one)
```bash
cd /Users/aneesh/Capstone/stage1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

#### Stage 1 Only (CLI)
```bash
python service.py "Login → Dashboard → Todos CRUD" --topk-components 8 --topk-routes 8
```

#### Stage 2A Only (CLI)
```bash
# From Stage 1 output
python subgraph_retrieval.py --input-json stage1_output.json --depth 2

# Or directly with routes/components
python subgraph_retrieval.py --routes /dashboard /login --components Dashboard Login --depth 2
```

#### Complete Pipeline (Stage 1 + Stage 2A)
```bash
# Using the example script
python example_stage1_2a.py "Register → Dashboard → Analytics"
```

#### Stage 3 Only (Test Generation)
```bash
# From pipeline output (Stage 1 + 2A)
python test_generator.py --input pipeline_output.json --output tests/scenario.spec.ts

# With custom framework
python test_generator.py --input pipeline_output.json --framework cypress --output tests/scenario.cy.js
```

#### Complete End-to-End Pipeline (Stage 1 + 2A + 3)
```bash
# Basic usage - generates Playwright test
python example_end_to_end.py "Login → Dashboard → Create Todo"

# With Cypress framework
python example_end_to_end.py "Register → Dashboard → Analytics" --framework cypress

# Save to specific file
python example_end_to_end.py "User workflow" --output ./tests/user-workflow.spec.ts

# Custom depth and framework
python example_end_to_end.py "Complex flow" --depth 3 --framework playwright --output ./tests/complex.spec.ts
```

#### Python API

**Stages 1 + 2A:**
```python
from stage1.service import Stage1Service
from stage1.subgraph_retrieval import retrieve_subgraph

# Stage 1: Entity extraction and interpretation
svc = Stage1Service()
stage1_result = svc.run_stage1("Login → Dashboard → Todos CRUD")

# Stage 2A: Subgraph retrieval
subgraph = retrieve_subgraph(stage1_result['structured'], depth=2)

print(f"Retrieved {subgraph['summary']['node_count']} nodes")
print(f"Retrieved {subgraph['summary']['edge_count']} edges")
```

**Complete Pipeline (Stages 1 + 2A + 3):**
```python
from stage1.service import Stage1Service
from stage1.subgraph_retrieval import retrieve_subgraph, Stage2Config
from stage1.test_generator import generate_test, Stage3Config, TestFramework

# Stage 1: Scenario interpretation
svc = Stage1Service()
stage1_result = svc.run_stage1("Login → Dashboard → Create Todo")

# Stage 2A: Subgraph retrieval with minimal filtering
config2 = Stage2Config()
config2.include_only_node_types = ["Component", "Route", "State", "EventHandler", "Hook", "Prop"]
subgraph = retrieve_subgraph(stage1_result['structured'], depth=2, config=config2)

# Stage 3: Test generation
config3 = Stage3Config()
config3.test_framework = TestFramework.PLAYWRIGHT
test_result = generate_test(stage1_result['structured'], subgraph, config=config3)

# Save generated test
with open("tests/todo-workflow.spec.ts", "w") as f:
    f.write(test_result["test_code"])

print(f"✅ Generated {test_result['framework']} test: {test_result['test_name']}")
print(f"   Assertions: {test_result['metadata']['assertions_count']}")
print(f"   Steps: {test_result['metadata']['steps_count']}")
```

### Notes
- If Neo4j vector indexes are unavailable, the service builds a local cache of node embeddings from `KGNode` properties: `name`, `description`, `embedding`.
- If a node lacks an `embedding`, one is computed from text using the configured embedding model.
- Vector index search uses `CALL db.index.vector.queryNodes('<index>', k, $embedding)`.


