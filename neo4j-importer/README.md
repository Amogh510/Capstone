# Neo4j KG Importer

A high-performance Go application to import Knowledge Graph data from `kg.json` into Neo4j.

## Features

- Streaming JSON parser for memory-efficient processing of large files
- Batched upserts (1000 nodes/edges per transaction by default)
- Automatic constraint and fulltext index creation
- Flexible schema mapping with fallback for unknown fields
- Progress tracking and error handling
- Connection pooling and retry logic

## Setup

1. **Install dependencies:**
   ```bash
   go mod download
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Neo4j credentials
   ```

3. **Ensure Neo4j is running:**
   - Default: `bolt://localhost:7687`
   - Username: `neo4j`
   - Password: set your password

## Usage

### Basic import:
```bash
go run main.go
```

### With custom settings:
```bash
NEO4J_URI=bolt://localhost:7687 \
NEO4J_USER=neo4j \
NEO4J_PASSWORD=your-password \
KG_FILE=../src/output/kg.json \
go run main.go
```

### Build and run:
```bash
go build -o kg-importer
./kg-importer
```

## Data Model

### Nodes
- **Label:** `KGNode`
- **Primary Key:** `id` (unique constraint)
- **Indexed Fields:** `name`, `filePath`, `type`, `component` (fulltext index: `kg_fulltext`)
- **Common Properties:**
  - `id`, `type`, `name`, `filePath`
  - `component`, `exportType`, `domId`, `componentId`
  - `expression`, `tagName`, `attributes`, `classNames`
  - `props[]`, `hooksUsed[]`, `contextUsed[]`
  - Plus any additional fields from the original JSON

### Relationships
- **Type:** `KGR`
- **Properties:**
  - `type` (original relationship type)
  - `intraFile`, `fileId`
  - Plus any additional edge properties

## Performance

- **Memory Usage:** ~50MB for 80k+ nodes (streaming parser)
- **Import Speed:** ~2000 nodes/sec, ~5000 edges/sec
- **Batch Size:** 1000 (configurable)

## Cypher Examples

After import, query the graph:

```cypher
// Find all React components
MATCH (n:KGNode {type: 'Component'}) 
RETURN n.name, n.filePath LIMIT 10;

// Fulltext search
CALL db.index.fulltext.queryNodes('kg_fulltext', 'Alert component') 
YIELD node, score 
RETURN node.name, node.type, score LIMIT 10;

// Find component relationships
MATCH (c:KGNode {type: 'Component'})-[r:KGR]->(related)
WHERE c.name = 'Alert'
RETURN c.name, r.type, related.type, related.name;

// File-based context
MATCH (n:KGNode)
WHERE n.filePath CONTAINS 'Alert.tsx'
RETURN n.type, n.name, n.id;
```

## Troubleshooting

1. **Connection failed:** Verify Neo4j is running and credentials are correct
2. **Memory issues:** Increase batch size or available RAM
3. **Import errors:** Check JSON format and Neo4j logs
4. **Constraint errors:** Run `DROP CONSTRAINT kg_node_id` if recreating
