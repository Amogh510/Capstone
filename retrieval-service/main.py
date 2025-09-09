"""
Knowledge Graph Retrieval Service for LLM Context
A FastAPI service that provides intelligent retrieval from Neo4j KG for UI codebases
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models for API requests/responses
class SearchType(str, Enum):
    FULLTEXT = "fulltext"
    COMPONENT = "component" 
    FILE = "file"
    SEMANTIC = "semantic"

class ContextRequest(BaseModel):
    query: str
    search_type: SearchType = SearchType.FULLTEXT
    max_results: int = 10
    expand_depth: int = 1
    include_code: bool = False
    file_filter: Optional[str] = None
    component_filter: Optional[str] = None

class KGNode(BaseModel):
    id: str
    type: str
    name: Optional[str] = None
    filePath: Optional[str] = None
    component: Optional[str] = None
    exportType: Optional[str] = None
    properties: Dict[str, Any] = {}

class KGRelationship(BaseModel):
    from_node: str
    to_node: str
    type: str
    properties: Dict[str, Any] = {}

class ContextResponse(BaseModel):
    query: str
    search_type: str
    results: List[KGNode]
    relationships: List[KGRelationship]
    context_summary: Dict[str, Any]
    llm_formatted_context: str

class AskRequest(BaseModel):
    question: str
    strategy: Optional[str] = "semantic"  # semantic | component | file
    component_name: Optional[str] = None
    file_path: Optional[str] = None
    max_results: int = 12
    expand_depth: int = 2
    model: Optional[str] = None  # override default model

class AskResponse(BaseModel):
    question: str
    answer: str
    retrieval: Dict[str, Any]

@dataclass
class RetrievalConfig:
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "admin123"
    max_context_length: int = 8000
    component_weight: float = 1.0
    file_weight: float = 0.8
    relationship_weight: float = 0.6
    embedding_model: str = "all-MiniLM-L6-v2"
    index_autoload: bool = True
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

class KnowledgeGraphRetriever:
    """Main retrieval service for the knowledge graph"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        logger.info(f"Connected to Neo4j at {config.neo4j_uri}")
        
        # Embedding model and in-memory index for components
        self._embed_lock = threading.Lock()
        self._model: Optional[SentenceTransformer] = None
        self._component_index: Dict[str, np.ndarray] = {}
        self._component_meta: Dict[str, Dict[str, Any]] = {}
        
        if self.config.index_autoload:
            # Load in background to keep startup fast
            threading.Thread(target=self._lazy_load_embeddings, daemon=True).start()

    def _lazy_load_embeddings(self):
        """Load model and precompute component embeddings."""
        try:
            logger.info("Loading embedding model %s...", self.config.embedding_model)
            with self._embed_lock:
                self._model = SentenceTransformer(self.config.embedding_model)
            logger.info("Embedding model loaded.")

            # Fetch components
            cypher = """
            MATCH (c:KGNode {type: 'Component'})
            RETURN c.id as id, c.name as name, c.filePath as filePath, c.exportType as exportType
            """
            records = self._execute_query(cypher)

            texts = []
            ids = []
            meta = []
            for rec in records:
                cid = rec.get("id")
                name = rec.get("name") or ""
                fp = rec.get("filePath") or ""
                export_type = rec.get("exportType") or ""
                text = f"Component: {name}\nFile: {fp}\nExport: {export_type}"
                ids.append(cid)
                texts.append(text)
                meta.append(rec)

            if not texts:
                logger.info("No component nodes found for embedding.")
                return

            logger.info("Computing embeddings for %d components...", len(texts))
            with self._embed_lock:
                embeddings = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

            # Save index
            for i, cid in enumerate(ids):
                self._component_index[cid] = embeddings[i]
                self._component_meta[cid] = meta[i]

            logger.info("Component embedding index ready (%d entries).", len(self._component_index))
        except Exception as e:
            logger.exception("Failed to build embedding index: %s", e)
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    
    def fulltext_search(self, query: str, limit: int = 10) -> List[KGNode]:
        """Search nodes using text matching (fallback for fulltext)"""
        cypher = """
        MATCH (n:KGNode)
        WHERE n.name CONTAINS $query 
           OR n.filePath CONTAINS $query
           OR n.component CONTAINS $query
        RETURN n
        ORDER BY 
            CASE 
                WHEN n.name CONTAINS $query THEN 1
                WHEN n.component CONTAINS $query THEN 2  
                ELSE 3
            END,
            n.name
        LIMIT $limit
        """
        
        results = self._execute_query(cypher, {"query": query, "limit": limit})
        return [self._record_to_node(record["n"]) for record in results]
    
    def component_search(self, component_name: str, limit: int = 10) -> List[KGNode]:
        """Find components and their related nodes"""
        cypher = """
        MATCH (c:KGNode {type: 'Component'})
        WHERE c.name CONTAINS $component_name
        OPTIONAL MATCH (c)-[r:KGR]->(related)
        RETURN c, collect(related) as related_nodes
        ORDER BY c.name
        LIMIT $limit
        """
        
        results = self._execute_query(cypher, {"component_name": component_name, "limit": limit})
        nodes = []
        
        for record in results:
            # Add the component itself
            nodes.append(self._record_to_node(record["c"]))
            # Add related nodes
            for related in record.get("related_nodes", []):
                if related:
                    nodes.append(self._record_to_node(related))
        
        return nodes
    
    def file_search(self, file_path: str, limit: int = 50) -> List[KGNode]:
        """Get all nodes from a specific file"""
        cypher = """
        MATCH (n:KGNode)
        WHERE n.filePath CONTAINS $file_path
        RETURN n
        ORDER BY 
            CASE n.type
                WHEN 'Component' THEN 1
                WHEN 'Hook' THEN 2
                WHEN 'State' THEN 3
                WHEN 'Prop' THEN 4
                ELSE 5
            END,
            n.name
        LIMIT $limit
        """
        
        results = self._execute_query(cypher, {"file_path": file_path, "limit": limit})
        return [self._record_to_node(record["n"]) for record in results]
    
    def expand_context(self, node_ids: List[str], depth: int = 1) -> tuple[List[KGNode], List[KGRelationship]]:
        """Expand context around given nodes by following relationships.
        Returns explicit relationship maps to avoid driver object handling issues.
        """
        if not node_ids or depth <= 0:
            return [], []

        if depth == 1:
            cypher = """
            MATCH (start:KGNode)
            WHERE start.id IN $node_ids
            MATCH (start)-[r:KGR]->(related)
            RETURN start, related, {from: start.id, to: related.id, type: r.type} AS rel
            LIMIT 1000
            """
            params = {"node_ids": node_ids}
        else:
            cypher = """
            MATCH (start:KGNode)
            WHERE start.id IN $node_ids
            MATCH p=(start)-[r:KGR*1..$depth]->(related)
            WITH start, related, relationships(p) AS rels
            UNWIND rels AS rel
            RETURN start, related, {from: startNode(rel).id, to: endNode(rel).id, type: rel.type} AS rel
            LIMIT 5000
            """
            params = {"node_ids": node_ids, "depth": depth}

        results = self._execute_query(cypher, params)

        nodes: List[KGNode] = []
        relationships: List[KGRelationship] = []
        seen_nodes: set[str] = set()

        for record in results:
            # Add start node
            start_node = self._record_to_node(record["start"])
            if start_node.id and start_node.id not in seen_nodes:
                nodes.append(start_node)
                seen_nodes.add(start_node.id)

            # Add related node
            related_node = self._record_to_node(record["related"])
            if related_node.id and related_node.id not in seen_nodes:
                nodes.append(related_node)
                seen_nodes.add(related_node.id)

            # Add relationship from explicit map
            rel_map = record.get("rel") or {}
            if isinstance(rel_map, dict):
                relationships.append(self._record_to_relationship(rel_map))

        return nodes, relationships
    
    def get_component_context(self, component_name: str, expand_depth: int = 1) -> tuple[List[KGNode], List[KGRelationship]]:
        """Get comprehensive context for a component"""
        # First find the component
        cypher = """
        MATCH (c:KGNode {type: 'Component'})
        WHERE c.name = $component_name OR c.name CONTAINS $component_name
        RETURN c
        LIMIT 5
        """
        
        results = self._execute_query(cypher, {"component_name": component_name})
        if not results:
            return [], []
        
        component_ids = [record["c"]["id"] for record in results]
        
        # Get the components themselves
        components = [self._record_to_node(record["c"]) for record in results]
        
        # Expand context around these components
        expanded_nodes, relationships = self.expand_context(component_ids, expand_depth)
        
        # Combine and deduplicate
        all_nodes = components + expanded_nodes
        seen_ids = set()
        unique_nodes = []
        for node in all_nodes:
            if node.id not in seen_ids:
                unique_nodes.append(node)
                seen_ids.add(node.id)
        
        return unique_nodes, relationships
    
    def search(self, request: ContextRequest) -> ContextResponse:
        """Main search interface"""
        nodes = []
        relationships = []
        
        # Primary search based on type
        if request.search_type == SearchType.FULLTEXT:
            nodes = self.fulltext_search(request.query, request.max_results)
        elif request.search_type == SearchType.COMPONENT:
            nodes, relationships = self.get_component_context(request.query, request.expand_depth)
        elif request.search_type == SearchType.FILE:
            nodes = self.file_search(request.query, request.max_results)
        elif request.search_type == SearchType.SEMANTIC:
            # Semantic component retrieval → context expansion → rerank
            nodes, relationships = self.semantic_component_retrieval(
                query=request.query,
                k=request.max_results,
                expand_depth=request.expand_depth,
            )
        
        # Apply filters
        if request.file_filter:
            nodes = [n for n in nodes if request.file_filter.lower() in (n.filePath or "").lower()]
        
        if request.component_filter:
            nodes = [n for n in nodes if request.component_filter.lower() in (n.component or "").lower()]
        
        # Expand context if needed and not already done
        if request.expand_depth > 0 and request.search_type not in [SearchType.COMPONENT]:
            node_ids = [node.id for node in nodes]
            expanded_nodes, expanded_rels = self.expand_context(node_ids, request.expand_depth)
            
            # Merge results
            all_nodes = nodes + expanded_nodes
            seen_ids = set()
            unique_nodes = []
            for node in all_nodes:
                if node.id not in seen_ids:
                    unique_nodes.append(node)
                    seen_ids.add(node.id)
            
            nodes = unique_nodes[:request.max_results * 2]  # Allow more results after expansion
            relationships.extend(expanded_rels)
        
        # Generate context summary
        context_summary = self._generate_context_summary(nodes, relationships)
        
        # Format for LLM
        llm_context = self._format_for_llm(nodes, relationships, request.query)
        
        return ContextResponse(
            query=request.query,
            search_type=request.search_type.value,
            results=nodes,
            relationships=relationships,
            context_summary=context_summary,
            llm_formatted_context=llm_context
        )

    def _ensure_model(self):
        if self._model is None:
            with self._embed_lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.config.embedding_model)

    def semantic_component_retrieval(self, query: str, k: int = 10, expand_depth: int = 1) -> Tuple[List[KGNode], List[KGRelationship]]:
        """Two-pass semantic retrieval restricted to components.
        1) Embed query and retrieve top-k similar components by cosine similarity
        2) Expand local context (neighbors) around those components
        3) Rerank expanded nodes with a heuristic favoring components and proximity
        """
        # Ensure index/model
        self._ensure_model()
        if not self._component_index:
            # If index not ready yet, build synchronously once
            self._lazy_load_embeddings()

        # Embed query
        with self._embed_lock:
            q_emb = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

        # Compute similarities
        sims: List[Tuple[str, float]] = []
        for cid, emb in self._component_index.items():
            score = float(np.dot(q_emb, emb))  # cosine since normalized
            sims.append((cid, score))
        sims.sort(key=lambda x: x[1], reverse=True)

        # Take top-k components
        top_component_ids = [cid for cid, _ in sims[:max(k, 5)]]

        # Fetch their nodes
        if not top_component_ids:
            return [], []

        cypher = """
        MATCH (c:KGNode {type: 'Component'})
        WHERE c.id IN $ids
        RETURN c
        """
        rows = self._execute_query(cypher, {"ids": top_component_ids})
        seed_nodes: List[KGNode] = [self._record_to_node(row["c"]) for row in rows]

        # Expand context around top components
        expanded_nodes, relationships = self.expand_context(top_component_ids, max(1, expand_depth))
        all_nodes = seed_nodes + expanded_nodes

        # Rerank nodes
        ranked = self._rerank_nodes(query, all_nodes, top_component_ids)
        top_nodes = [n for n, _ in ranked[:k]]

        return top_nodes, relationships

    def _rerank_nodes(self, query: str, nodes: List[KGNode], seed_ids: List[str]) -> List[Tuple[KGNode, float]]:
        """Heuristic reranker combining semantic similarity (title/text) and node type/seed proximity."""
        with self._embed_lock:
            q_emb = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

        scored: List[Tuple[KGNode, float]] = []
        for n in nodes:
            # Compose a short text representation
            text = f"{n.type}: {n.name or ''} {n.filePath or ''}"
            with self._embed_lock:
                n_emb = self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            sim = float(np.dot(q_emb, n_emb))

            # Type boost
            type_boost = 0.0
            if n.type == "Component":
                type_boost += 0.2
            elif n.type in ("Prop", "Hook", "State"):
                type_boost += 0.1

            # Seed proximity boost
            proximity_boost = 0.2 if n.id in seed_ids else 0.0

            score = sim + type_boost + proximity_boost
            scored.append((n, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ---------- LLM (Ollama) layer ----------
    def _build_graph_bundle(self, nodes: List[KGNode], relationships: List[KGRelationship]) -> Dict[str, Any]:
        by_file: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            fp = n.filePath or "unknown"
            entry = by_file.setdefault(fp, {"components": [], "hooks": [], "states": [], "props": [], "jsx": [], "other": []})
            item = {"id": n.id, "type": n.type, "name": n.name, "exportType": n.exportType}
            if n.type == "Component":
                entry["components"].append(item)
            elif n.type == "Hook":
                entry["hooks"].append(item)
            elif n.type == "State":
                entry["states"].append(item)
            elif n.type == "Prop":
                entry["props"].append(item)
            elif n.type == "JSXElement":
                entry["jsx"].append(item)
            else:
                entry["other"].append(item)

        edges = [{"from": r.from_node, "to": r.to_node, "type": r.type} for r in relationships]
        return {"files": by_file, "edges": edges}

    def _compose_ollama_prompt(self, question: str, nodes: List[KGNode], relationships: List[KGRelationship]) -> str:
        graph_bundle = self._build_graph_bundle(nodes, relationships)
        bundle_json = json.dumps(graph_bundle, indent=2)[:12000]  # cap prompt size
        context_text = self._format_for_llm(nodes, relationships, question)
        prompt = (
            "You are a senior UI engineer. You are given a knowledge graph extracted from a large React UI codebase.\n"
            "Use the provided graph bundle and summary to answer the question precisely.\n"
            "- Prioritize Components and their relationships (props, hooks, jsx, routes).\n"
            "- When relevant, list key files and components.\n"
            "- Reference component names and file paths.\n"
            "- If multiple relevant components exist, rank them and explain why.\n"
            "- Suggest next code files to inspect if needed.\n\n"
            f"Question:\n{question}\n\n"
            f"Graph Summary:\n{context_text}\n\n"
            f"Graph Bundle (JSON):\n{bundle_json}\n\n"
            "Answer in a concise, structured way with bullet points and short explanations."
        )
        return prompt

    def _call_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        base = self.config.ollama_base_url.rstrip('/')
        use_model = model or self.config.ollama_model
        try:
            resp = requests.post(
                f"{base}/api/generate",
                json={"model": use_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

    def ask(self, req: 'AskRequest') -> 'AskResponse':
        # Retrieval strategy
        if req.strategy == "component" and req.component_name:
            nodes, rels = self.get_component_context(req.component_name, req.expand_depth)
        elif req.strategy == "file" and req.file_path:
            nodes = self.file_search(req.file_path, req.max_results)
            node_ids = [n.id for n in nodes]
            extra_nodes, rels = self.expand_context(node_ids, req.expand_depth)
            nodes = self._unique_nodes(nodes + extra_nodes)
        else:
            nodes, rels = self.semantic_component_retrieval(req.question, req.max_results, req.expand_depth)

        prompt = self._compose_ollama_prompt(req.question, nodes, rels)
        answer = self._call_ollama(prompt, model=req.model)

        retrieval_info = {
            "total_nodes": len(nodes),
            "total_relationships": len(rels),
            "top_files": list({n.filePath for n in nodes if n.filePath})[:10],
            "top_components": [n.name for n in nodes if n.type == "Component"][:10],
        }
        return AskResponse(question=req.question, answer=answer, retrieval=retrieval_info)

    def _unique_nodes(self, nodes: List[KGNode]) -> List[KGNode]:
        seen = set()
        out: List[KGNode] = []
        for n in nodes:
            if n.id not in seen:
                out.append(n)
                seen.add(n.id)
        return out
    
    def _record_to_node(self, record: Dict[str, Any]) -> KGNode:
        """Convert Neo4j record to KGNode"""
        node_data = dict(record)
        
        return KGNode(
            id=node_data.get("id", ""),
            type=node_data.get("type", ""),
            name=node_data.get("name"),
            filePath=node_data.get("filePath"),
            component=node_data.get("component"),
            exportType=node_data.get("exportType"),
            properties={k: v for k, v in node_data.items() 
                       if k not in ["id", "type", "name", "filePath", "component", "exportType"]}
        )
    
    def _record_to_relationship(self, record: Dict[str, Any]) -> KGRelationship:
        """Convert Neo4j relationship record to KGRelationship"""
        return KGRelationship(
            from_node=record.get("from", ""),
            to_node=record.get("to", ""),
            type=record.get("type", ""),
            properties={k: v for k, v in record.items() 
                       if k not in ["from", "to", "type"]}
        )
    
    def _generate_context_summary(self, nodes: List[KGNode], relationships: List[KGRelationship]) -> Dict[str, Any]:
        """Generate a summary of the retrieved context"""
        node_types = {}
        files = set()
        components = set()
        
        for node in nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
            if node.filePath:
                files.add(node.filePath)
            if node.component:
                components.add(node.component)
        
        rel_types = {}
        for rel in relationships:
            rel_types[rel.type] = rel_types.get(rel.type, 0) + 1
        
        return {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "node_types": node_types,
            "relationship_types": rel_types,
            "files_involved": list(files)[:10],  # Limit for readability
            "components_involved": list(components)[:10],
            "coverage": {
                "files": len(files),
                "components": len(components)
            }
        }
    
    def _format_for_llm(self, nodes: List[KGNode], relationships: List[KGRelationship], query: str) -> str:
        """Format the context in an LLM-friendly way"""
        
        context_parts = []
        
        # Header
        context_parts.append(f"# Codebase Context for Query: '{query}'\n")
        
        # Summary
        summary = self._generate_context_summary(nodes, relationships)
        context_parts.append(f"## Summary")
        context_parts.append(f"- Found {summary['total_nodes']} nodes across {summary['coverage']['files']} files")
        context_parts.append(f"- {summary['coverage']['components']} components involved")
        context_parts.append(f"- {summary['total_relationships']} relationships\n")
        
        # Group nodes by file for better organization
        nodes_by_file = {}
        for node in nodes:
            file_path = node.filePath or "unknown"
            if file_path not in nodes_by_file:
                nodes_by_file[file_path] = []
            nodes_by_file[file_path].append(node)
        
        # Format each file's content
        for file_path, file_nodes in nodes_by_file.items():
            if file_path == "unknown":
                continue
                
            context_parts.append(f"## File: {file_path}")
            
            # Group by type within file
            components = [n for n in file_nodes if n.type == "Component"]
            hooks = [n for n in file_nodes if n.type == "Hook"]
            states = [n for n in file_nodes if n.type == "State"]
            props = [n for n in file_nodes if n.type == "Prop"]
            jsx_elements = [n for n in file_nodes if n.type == "JSXElement"]
            others = [n for n in file_nodes if n.type not in ["Component", "Hook", "State", "Prop", "JSXElement"]]
            
            if components:
                context_parts.append("### Components:")
                for comp in components:
                    context_parts.append(f"- **{comp.name}** ({comp.exportType or 'unknown export'})")
                    if comp.properties.get("props"):
                        context_parts.append(f"  - Props: {comp.properties['props']}")
                    if comp.properties.get("hooksUsed"):
                        context_parts.append(f"  - Hooks: {comp.properties['hooksUsed']}")
            
            if hooks:
                context_parts.append("### Hooks:")
                for hook in hooks:
                    context_parts.append(f"- {hook.name}")
            
            if states:
                context_parts.append("### State:")
                for state in states:
                    context_parts.append(f"- {state.name}")
            
            if jsx_elements:
                context_parts.append("### JSX Elements:")
                jsx_summary = {}
                for jsx in jsx_elements:
                    tag = jsx.properties.get("tagName", "unknown")
                    jsx_summary[tag] = jsx_summary.get(tag, 0) + 1
                for tag, count in jsx_summary.items():
                    context_parts.append(f"- {tag}: {count} instances")
            
            if others:
                context_parts.append("### Other:")
                for other in others:
                    context_parts.append(f"- {other.type}: {other.name or other.id}")
            
            context_parts.append("")  # Empty line between files
        
        # Add key relationships
        if relationships:
            context_parts.append("## Key Relationships:")
            rel_summary = {}
            for rel in relationships:
                rel_type = rel.type
                rel_summary[rel_type] = rel_summary.get(rel_type, 0) + 1
            
            for rel_type, count in sorted(rel_summary.items(), key=lambda x: x[1], reverse=True):
                context_parts.append(f"- {rel_type}: {count} connections")
        
        # Truncate if too long
        full_context = "\n".join(context_parts)
        if len(full_context) > self.config.max_context_length:
            truncated = full_context[:self.config.max_context_length]
            last_newline = truncated.rfind('\n')
            if last_newline > 0:
                full_context = truncated[:last_newline] + "\n\n[Context truncated for length]"
        
        return full_context

# Initialize the retrieval service
config = RetrievalConfig(
    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
    neo4j_password=os.getenv("NEO4J_PASSWORD", "admin123")
)

retriever = KnowledgeGraphRetriever(config)

# FastAPI app
app = FastAPI(
    title="Knowledge Graph Retrieval Service",
    description="Intelligent retrieval system for UI codebase knowledge graphs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
def shutdown_event():
    retriever.close()

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Knowledge Graph Retrieval Service", "version": "1.0.0"}

@app.get("/health")
def health_check():
    # Report embedding index readiness
    index_ready = isinstance(retriever._model, SentenceTransformer) and len(retriever._component_index) > 0
    return {"status": "healthy", "neo4j_connected": True, "embedding_index_ready": index_ready}

@app.post("/search", response_model=ContextResponse)
def search_context(request: ContextRequest):
    """Main search endpoint for retrieving context"""
    try:
        return retriever.search(request)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/component/{component_name}")
def search_component(
    component_name: str,
    expand_depth: int = Query(1, ge=0, le=3),
    max_results: int = Query(20, ge=1, le=100)
):
    """Quick component search endpoint"""
    request = ContextRequest(
        query=component_name,
        search_type=SearchType.COMPONENT,
        expand_depth=expand_depth,
        max_results=max_results
    )
    return retriever.search(request)

@app.get("/search/file")
def search_file(
    file_path: str = Query(..., description="File path to search"),
    max_results: int = Query(50, ge=1, le=200)
):
    """Quick file search endpoint"""
    request = ContextRequest(
        query=file_path,
        search_type=SearchType.FILE,
        max_results=max_results
    )
    return retriever.search(request)

@app.get("/search/fulltext")
def search_fulltext(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=50),
    expand_depth: int = Query(0, ge=0, le=2)
):
    """Quick fulltext search endpoint"""
    request = ContextRequest(
        query=q,
        search_type=SearchType.FULLTEXT,
        max_results=max_results,
        expand_depth=expand_depth
    )
    return retriever.search(request)

@app.get("/search/semantic")
def search_semantic(
    q: str = Query(..., description="Search query (components only)"),
    max_results: int = Query(10, ge=1, le=50),
    expand_depth: int = Query(1, ge=0, le=3)
):
    request = ContextRequest(
        query=q,
        search_type=SearchType.SEMANTIC,
        max_results=max_results,
        expand_depth=expand_depth
    )
    return retriever.search(request)

@app.get("/stats")
def get_stats():
    """Get database statistics"""
    try:
        stats_query = """
        MATCH (n:KGNode)
        RETURN n.type as node_type, count(*) as count
        ORDER BY count DESC
        """
        node_stats = retriever._execute_query(stats_query)
        
        rel_query = """
        MATCH ()-[r:KGR]->()
        RETURN r.type as rel_type, count(*) as count
        ORDER BY count DESC
        """
        rel_stats = retriever._execute_query(rel_query)
        
        return {
            "node_statistics": node_stats,
            "relationship_statistics": rel_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask_codebase(req: AskRequest):
    """High-level QA endpoint powered by Ollama over the retrieved graph context."""
    try:
        return retriever.ask(req)
    except Exception as e:
        logger.error("/ask failed: %s", e)
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


