# Unified KG Pipeline Orchestrator

This repository now includes a single command to analyze a UI codebase, import the generated Knowledge Graph into Neo4j, and start the retrieval service.

## Prerequisites
- Node.js and npm
- Go
- Python 3 + pip3
- Neo4j running locally (bolt://localhost:7687) with credentials in env (defaults below)
- Optional: cypher-shell (improves DB clean step)

Configure environment (optional; defaults shown):
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=admin123
```

## One-shot run
```bash
cd Capstone
python3 run_pipeline.py /absolute/path/to/your/codebase/src
```

What it does:
1. Installs dependencies for Go and Python (and root npm if present)
2. Updates `src/config.js` to point `projectRoot` at your provided src
3. Runs the Node analyzer (writes `src/output/kg.json` and others)
4. Cleans Neo4j (fresh DB) and imports the KG via the Go importer
5. Starts the retrieval service at `http://localhost:8000` and waits until healthy

Progress bars are printed for each stage.

## Notes
- If `cypher-shell` is not available, it will use the Python Neo4j driver to clean the DB.
- The Go importer reads `KG_FILE` from `src/output/kg.json`.
- The retrieval service autoloads a component embedding index in the background.

## Troubleshooting
- Ensure Neo4j is running and reachable.
- If the analyzer fails, confirm your provided source path exists and is a React/TS project.
- If the retrieval service does not become healthy, check logs in the console and verify credentials.
