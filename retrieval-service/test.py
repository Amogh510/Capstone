"""
Knowledge Graph Retrieval Service for LLM Context (Upgraded)
- LLM-guided retrieval planning
- Hybrid candidate pooling (semantic + keyword + file + component hints)
- Graph-aware re-ranking (hop distance + edge-type weights + type boosts)
- Optional Cross-Encoder re-ranking
- Extended KG schema mapping (domId, tagName, hooksUsed, props, etc.)
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

# CrossEncoder is optional
try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ----------------------------- Setup -----------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kg-retrieval")

# ------------------------- API Models ----------------------------

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
    # Extended to mirror your Go struct
    id: str
    type: str
    name: Optional[str] = None
    filePath: Optional[str] = None
    component: Optional[str] = None
    exportType: Optional[str] = None
    domId: Optional[str] = None
    componentId: Optional[str] = None
    expression: Optional[str] = None
    tagName: Optional[str] = None
    attributes: Dict[str, Any] = {}
    classNames: List[str] = []
    props: List[str] = []
    hooksUsed: List[str] = []
    contextUsed: List[str] = []
    properties: Dict[str, Any] = {}  # carry any other fields

class KGRelationship(BaseModel):
    from_node: str
    to_node: str
    type: str
    intraFile: Optional[bool] = None
    fileId: Optional[str] = None
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

# ------------------------- Config -------------------------------

@dataclass
class RetrievalConfig:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "admin123")

    max_context_length: int = 8000

    embedding_model: str = "all-MiniLM-L6-v2"
    index_autoload: bool = True

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
    llm_planner_model: Optional[str] = os.getenv("RETRIEVAL_LLM_PLANNER_MODEL", None)

    # Cross-encoder re-ranking
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_cross_encoder: bool = True

    # Graph-aware scoring
    hop_penalty: float = 0.15  # subtract per hop
    edge_type_weights: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.edge_type_weights is None:
            # Tune against your schema/distribution
            self.edge_type_weights = {
                "USES_HOOK": 0.25,
                "HAS_PROP": 0.20,
                "RENDERS": 0.20,
                "IMPORTS": 0.15,
                "ROUTES_TO": 0.15,
                "MENTIONS": 0.05,
                "KGR": 0.05,  # generic fallback
            }

# ----------------------- Retriever Core -------------------------

class KnowledgeGraphRetriever:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_password)
        )
        logger.info("Connected to Neo4j at %s", config.neo4j_uri)

        # Embedding model + component index
        self._embed_lock = threading.Lock()
        self._model: Optional[SentenceTransformer] = None
        self._component_index: Dict[str, np.ndarray] = {}
        self._component_meta: Dict[str, Dict[str, Any]] = {}

        # Cross-encoder
        self._cross: Optional[CrossEncoder] = None

        if self.config.index_autoload:
            threading.Thread(target=self._lazy_load_embeddings, daemon=True).start()

    # ---------------- Neo4j Helpers ----------------

    def close(self):
        self.driver.close()

    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
            except Exception as e:
                logger.error("Query execution failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    # ---------------- Embeddings -------------------

    def _ensure_model(self):
        if self._model is None:
            with self._embed_lock:
                if self._model is None:
                    logger.info("Loading embedding model: %s", self.config.embedding_model)
                    self._model = SentenceTransformer(self.config.embedding_model)
                    logger.info("Embedding model loaded.")

    def _ensure_cross(self):
        if not self.config.use_cross_encoder:
            return
        if CrossEncoder is None:
            logger.warning("CrossEncoder not installed; disabling cross re-rank.")
            self.config.use_cross_encoder = False
            return
        if self._cross is None:
            try:
                self._cross = CrossEncoder(self.config.cross_encoder_model)
                logger.info("Loaded cross-encoder: %s", self.config.cross_encoder_model)
            except Exception as e:
                logger.warning("Failed to load cross-encoder; disabling. %s", e)
                self.config.use_cross_encoder = False

    def _lazy_load_embeddings(self):
        try:
            self._ensure_model()

            # Fetch components to index
            cypher = """
            MATCH (c:KGNode {type: 'Component'})
            RETURN c.id AS id, c.name AS name, c.filePath AS filePath, c.exportType AS exportType
            """
            records = self._execute_query(cypher)

            ids, texts, meta = [], [], []
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

            with self._embed_lock:
                emb = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

            for i, cid in enumerate(ids):
                self._component_index[cid] = emb[i]
                self._component_meta[cid] = meta[i]

            logger.info("Component embedding index ready (%d entries).", len(self._component_index))
        except Exception as e:
            logger.exception("Failed to build embedding index: %s", e)

    # ---------------- Search Strategies ------------

    def fulltext_search(self, query: str, limit: int = 10) -> List[KGNode]:
        cypher = """
        MATCH (n:KGNode)
        WHERE (n.name IS NOT NULL AND n.name CONTAINS $query)
           OR (n.filePath IS NOT NULL AND n.filePath CONTAINS $query)
           OR (n.component IS NOT NULL AND n.component CONTAINS $query)
        RETURN n
        ORDER BY 
            CASE 
                WHEN n.name IS NOT NULL AND n.name CONTAINS $query THEN 1
                WHEN n.component IS NOT NULL AND n.component CONTAINS $query THEN 2
                ELSE 3
            END, n.name
        LIMIT $limit
        """
        results = self._execute_query(cypher, {"query": query, "limit": limit})
        return [self._record_to_node(record["n"]) for record in results]

    def component_search(self, component_name: str, limit: int = 10) -> List[KGNode]:
        cypher = """
        MATCH (c:KGNode {type: 'Component'})
        WHERE c.name CONTAINS $component_name
        OPTIONAL MATCH (c)-[r:KGR]->(related)
        RETURN c, collect(related) AS related_nodes
        ORDER BY c.name
        LIMIT $limit
        """
        results = self._execute_query(cypher, {"component_name": component_name, "limit": limit})
        nodes: List[KGNode] = []
        for rec in results:
            nodes.append(self._record_to_node(rec["c"]))
            for reln in rec.get("related_nodes", []):
                if reln:
                    nodes.append(self._record_to_node(reln))
        return nodes

    def file_search(self, file_path: str, limit: int = 50) -> List[KGNode]:
        cypher = """
        MATCH (n:KGNode)
        WHERE n.filePath IS NOT NULL AND n.filePath CONTAINS $file_path
        RETURN n
        ORDER BY 
            CASE n.type
                WHEN 'Component' THEN 1
                WHEN 'Hook' THEN 2
                WHEN 'State' THEN 3
                WHEN 'Prop' THEN 4
                ELSE 5
            END, n.name
        LIMIT $limit
        """
        results = self._execute_query(cypher, {"file_path": file_path, "limit": limit})
        return [self._record_to_node(record["n"]) for record in results]

    def expand_context(self, node_ids: List[str], depth: int = 1) -> Tuple[List[KGNode], List[KGRelationship]]:
        if not node_ids or depth <= 0:
            return [], []

        if depth == 1:
            cypher = """
            MATCH (start:KGNode)
            WHERE start.id IN $node_ids
            MATCH (start)-[r:KGR]->(related)
            RETURN start, related, {from: start.id, to: related.id, type: r.type, intraFile: r.intraFile, fileId: r.fileId} AS rel
            LIMIT 3000
            """
            params = {"node_ids": node_ids}
        else:
            # Neo4j doesn't allow parameterizing variable-length patterns; inline the bounded depth integer
            cypher = f"""
            MATCH (start:KGNode)
            WHERE start.id IN $node_ids
            MATCH p=(start)-[r:KGR*1..{int(depth)}]->(related)
            WITH start, related, relationships(p) AS rels
            UNWIND rels AS rel
            RETURN start, related, {{from: startNode(rel).id, to: endNode(rel).id, type: rel.type, intraFile: rel.intraFile, fileId: rel.fileId}} AS rel
            LIMIT 8000
            """
            params = {"node_ids": node_ids}

        rows = self._execute_query(cypher, params)

        nodes: List[KGNode] = []
        rels: List[KGRelationship] = []
        seen_ids: set = set()

        for row in rows:
            s = self._record_to_node(row["start"])
            if s.id and s.id not in seen_ids:
                nodes.append(s); seen_ids.add(s.id)

            t = self._record_to_node(row["related"])
            if t.id and t.id not in seen_ids:
                nodes.append(t); seen_ids.add(t.id)

            rels.append(self._record_to_relationship(row.get("rel") or {}))

        return nodes, rels

    # ---------------- Semantic Retrieval ------------

    def _embed_text(self, text: str) -> np.ndarray:
        with self._embed_lock:
            return self._model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]

    def semantic_component_retrieval(self, query: str, k: int = 10, expand_depth: int = 1) -> Tuple[List[KGNode], List[KGRelationship]]:
        self._ensure_model()
        if not self._component_index:
            self._lazy_load_embeddings()

        q_emb = self._embed_text(query)

        sims: List[Tuple[str, float]] = []
        for cid, emb in self._component_index.items():
            sims.append((cid, float(np.dot(q_emb, emb))))
        sims.sort(key=lambda x: x[1], reverse=True)

        top_ids = [cid for cid, _ in sims[:max(k, 5)]]
        if not top_ids:
            return [], []

        cypher = "MATCH (c:KGNode {type: 'Component'}) WHERE c.id IN $ids RETURN c"
        rows = self._execute_query(cypher, {"ids": top_ids})
        seeds = [self._record_to_node(r["c"]) for r in rows]

        expanded_nodes, rels = self.expand_context(top_ids, max(1, expand_depth))
        all_nodes = self._unique_nodes(seeds + expanded_nodes)

        ranked = self._rerank_nodes_basic(query, all_nodes, top_ids)
        return [n for n, _ in ranked[:k]], rels

    # ---------------- Re-ranking --------------------

    def _rerank_nodes_basic(self, query: str, nodes: List[KGNode], seed_ids: List[str]) -> List[Tuple[KGNode, float]]:
        self._ensure_model()
        q_emb = self._embed_text(query)
        scored: List[Tuple[KGNode, float]] = []
        for n in nodes:
            text = f"{n.type}: {n.name or ''} {n.filePath or ''}"
            n_emb = self._embed_text(text)
            sim = float(np.dot(q_emb, n_emb))
            boost = 0.2 if n.type == "Component" else (0.1 if n.type in ("Prop", "Hook", "State") else 0.0)
            prox = 0.2 if n.id in seed_ids else 0.0
            scored.append((n, sim + boost + prox))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_min_hops(self, seed_ids: List[str], nodes: List[KGNode], rels: List[KGRelationship]) -> Dict[str, int]:
        adj: Dict[str, List[str]] = {}
        for r in rels:
            adj.setdefault(r.from_node, []).append(r.to_node)
            adj.setdefault(r.to_node, []).append(r.from_node)  # treat as undirected

        from collections import deque
        dist = {sid: 0 for sid in seed_ids}
        dq = deque(seed_ids)
        seen = set(seed_ids)

        while dq:
            cur = dq.popleft()
            for nb in adj.get(cur, []):
                if nb not in seen:
                    seen.add(nb)
                    dist[nb] = dist[cur] + 1
                    dq.append(nb)

        default = 3
        return {n.id: dist.get(n.id, default) for n in nodes}

    def _edge_weight_score(self, node_id: str, rels: List[KGRelationship]) -> float:
        """Sum weights of incident edges (1-hop) based on edge type."""
        w = 0.0
        weights = self.config.edge_type_weights or {}
        for r in rels:
            if r.from_node == node_id or r.to_node == node_id:
                w += weights.get(r.type, weights.get("KGR", 0.0))
        # dampen to a small bonus range
        return min(0.4, w * 0.05)

    def _rerank_nodes_graphaware(
        self,
        query: str,
        nodes: List[KGNode],
        seed_ids: List[str],
        rels: List[KGRelationship],
        top_n: int
    ) -> List[KGNode]:
        self._ensure_model()
        q_emb = self._embed_text(query)

        # Bi-encoder scores
        bi = {}
        for n in nodes:
            text = f"{n.type}: {n.name or ''} {n.filePath or ''}"
            bi[n.id] = float(np.dot(q_emb, self._embed_text(text)))

        # Optional cross-encoder
        self._ensure_cross()
        cross: Dict[str, float] = {}
        if self.config.use_cross_encoder and self._cross is not None:
            pairs = [(query, f"{n.type}: {n.name or ''} {n.filePath or ''}") for n in nodes]
            try:
                scores = self._cross.predict(pairs, convert_to_numpy=True)
                for n, s in zip(nodes, scores):
                    cross[n.id] = float(s)
            except Exception as e:
                logger.warning("Cross re-rank failed; continuing without. %s", e)
                cross = {}

        hops = self._compute_min_hops(seed_ids, nodes, rels)

        def type_boost(t: str) -> float:
            # Strongly favor Components; mildly favor Prop/Hook/State; de-emphasize styling-only nodes
            if t == "Component":
                return 0.6
            if t in ("Prop", "Hook", "State"):
                return 0.2
            if t in ("TailwindUtility",):
                return -0.3
            if t in ("JSXElement",):
                return -0.2
            return 0.0

        def seed_bonus(i: str) -> float:
            return 0.4 if i in seed_ids else 0.0

        alpha, beta = (0.6, 0.4) if cross else (1.0, 0.0)

        scored: List[Tuple[KGNode, float]] = []
        for n in nodes:
            score = alpha * bi.get(n.id, 0.0) + beta * cross.get(n.id, 0.0)
            score += type_boost(n.type)
            score += seed_bonus(n.id)
            score -= self.config.hop_penalty * hops.get(n.id, 3)
            score += self._edge_weight_score(n.id, rels)
            scored.append((n, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:top_n]]

    # ---------------- Planner / Hybrid ----------------

    def _ensure_list_of_strings(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, (str, int, float)):
                    s = str(item).strip()
                    if s:
                        out.append(s)
            return out
        # unsupported types -> empty
        return []

    def _extract_flow_pair(self, question: str, plan: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Try to detect a source→target flow pair like 'from Login to Analytics'.
        Falls back to splitting planner terms by comma/and.
        """
        q = question.strip()
        m = re.search(r"from\s+(.+?)\s+to\s+(.+?)([?.,;]|$)", q, flags=re.IGNORECASE)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            if a and b and a.lower() != b.lower():
                return a, b
        # fallback: try to split the first term that contains a comma/and
        candidates: List[str] = []
        for arr_key in ["must_terms", "synonyms", "component_hints", "file_hints"]:
            arr = plan.get(arr_key) or []
            for s in arr:
                parts = re.split(r"\band\b|,", s, flags=re.IGNORECASE)
                for p in parts:
                    p = p.strip()
                    if p:
                        candidates.append(p)
            if len(candidates) >= 2:
                break
        if len(candidates) >= 2 and candidates[0].lower() != candidates[1].lower():
            return candidates[0], candidates[1]
        return None

    def _find_component_ids_by_name(self, name: str, limit: int = 5) -> List[str]:
        cypher = """
        MATCH (c:KGNode {type: 'Component'})
        WHERE toLower(c.name) = toLower($name)
           OR (exists(c.name) AND toLower(c.name) CONTAINS toLower($name))
        RETURN c.id AS id
        LIMIT $limit
        """
        # exists(c.name) - adjust for Neo4j 5 IS NOT NULL
        cypher = cypher.replace("exists(c.name)", "c.name IS NOT NULL")
        rows = self._execute_query(cypher, {"name": name, "limit": limit})
        return [r.get("id") for r in rows if r.get("id")]

    def _find_paths_between_components(
        self,
        source_ids: List[str],
        target_ids: List[str],
        max_len: int = 4,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between any of the provided component ids.
        Returns a list of paths, each with nodes and edges in order.
        """
        paths: List[Dict[str, Any]] = []
        for sid in source_ids[:5]:
            for tid in target_ids[:5]:
                if sid == tid:
                    continue
                cypher = f"""
                MATCH (a:KGNode {id: $sid}), (b:KGNode {id: $tid})
                MATCH p=shortestPath((a)-[:KGR*..{int(max_len)}]-(b))
                RETURN p
                LIMIT {int(limit)}
                """
                rows = self._execute_query(cypher, {"sid": sid, "tid": tid})
                for row in rows:
                    p = row.get("p")
                    if not p:
                        continue
                    # Neo4j's driver returns paths as objects; our helper expects dicts
                    # We'll re-run a query to extract nodes and rels in a JSON-friendly shape.
                    cy2 = f"""
                    MATCH (a:KGNode {{id: $sid}}), (b:KGNode {{id: $tid}})
                    MATCH p=shortestPath((a)-[:KGR*..{int(max_len)}]-(b))
                    WITH p
                    RETURN [n IN nodes(p) | n] AS ns,
                           [r IN relationships(p) | {{from: startNode(r).id, to: endNode(r).id, type: r.type, intraFile: r.intraFile, fileId: r.fileId}}] AS rs
                    LIMIT 1
                    """
                    parts = self._execute_query(cy2, {"sid": sid, "tid": tid})
                    if parts:
                        rec = parts[0]
                        ns = [self._record_to_node(n) for n in rec.get("ns", [])]
                        rs = [self._record_to_relationship(r) for r in rec.get("rs", [])]
                        paths.append({
                            "source_id": sid,
                            "target_id": tid,
                            "nodes": [n.dict() for n in ns],
                            "edges": [{
                                "from": r.from_node, "to": r.to_node, "type": r.type,
                                "intraFile": r.intraFile, "fileId": r.fileId
                            } for r in rs]
                        })
        return paths[:limit]

    def _call_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        base = self.config.ollama_base_url.rstrip("/")
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

    def _planner(self, question: str) -> Dict[str, Any]:
        prompt = (
            "You are a retrieval planner for a React UI knowledge graph.\n"
            "Given a user question, output ONLY valid JSON with keys:\n"
            "must_terms: string[]\n"
            "synonyms: string[]\n"
            "component_hints: string[]\n"
            "file_hints: string[]\n"
            "edge_types: string[]\n"
            "expand_depth: number\n"
            "negatives: string[]\n\n"
            f"Question: {question}\nJSON:"
        )
        try:
            model = self.config.llm_planner_model or self.config.ollama_model
            raw = self._call_ollama(prompt, model=model)
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1:
                obj = json.loads(raw[s:e+1])
                # sanitize
                obj.setdefault("must_terms", [])
                obj.setdefault("synonyms", [])
                obj.setdefault("component_hints", [])
                obj.setdefault("file_hints", [])
                obj.setdefault("edge_types", [])
                obj.setdefault("expand_depth", 2)
                # coerce expand_depth to a small integer range 0..3
                try:
                    ed = int(obj.get("expand_depth", 2))
                except (ValueError, TypeError):
                    ed = 2
                obj["expand_depth"] = max(0, min(3, ed))
                obj.setdefault("negatives", [])
                # coerce list-like fields to list[str]
                for key in [
                    "must_terms",
                    "synonyms",
                    "component_hints",
                    "file_hints",
                    "edge_types",
                    "negatives",
                ]:
                    obj[key] = self._ensure_list_of_strings(obj.get(key))
                return obj
        except Exception as e:
            logger.warning("Planner failed, using defaults. %s", e)
        return {"must_terms": [], "synonyms": [], "component_hints": [],
                "file_hints": [], "edge_types": [], "expand_depth": 2, "negatives": []}

    def _hybrid_candidates(self, plan: Dict[str, Any], query: str, k_sem: int = 20) -> List[KGNode]:
        pool: Dict[str, KGNode] = {}

        # 1) semantic components
        sem_nodes, _ = self.semantic_component_retrieval(query, k=k_sem, expand_depth=1)
        for n in sem_nodes:
            pool[n.id] = n

        # 2) full-text terms (must + synonyms)
        for t in (plan.get("must_terms", []) + plan.get("synonyms", []))[:5]:
            for n in self.fulltext_search(t, limit=10):
                pool[n.id] = n

        # 3) file hints
        for f in plan.get("file_hints", [])[:3]:
            for n in self.file_search(f, limit=30):
                pool[n.id] = n

        # 4) explicit component hints
        for c in plan.get("component_hints", [])[:3]:
            nodes, _r = self.get_component_context(c, expand_depth=1)
            for n in nodes:
                pool[n.id] = n

        return list(pool.values())

    # ---------------- LLM Context Formatting ---------

    def _build_graph_bundle(self, nodes: List[KGNode], relationships: List[KGRelationship]) -> Dict[str, Any]:
        by_file: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            fp = n.filePath or "unknown"
            entry = by_file.setdefault(fp, {
                "components": [], "hooks": [], "states": [], "props": [], "jsx": [], "other": []
            })
            item = {
                "id": n.id, "type": n.type, "name": n.name, "exportType": n.exportType,
                "domId": n.domId, "componentId": n.componentId, "tagName": n.tagName,
                "props": n.props, "hooksUsed": n.hooksUsed, "contextUsed": n.contextUsed,
            }
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

        edges = [{
            "from": r.from_node, "to": r.to_node, "type": r.type,
            "intraFile": r.intraFile, "fileId": r.fileId
        } for r in relationships]
        return {"files": by_file, "edges": edges}

    def _format_for_llm(self, nodes: List[KGNode], relationships: List[KGRelationship], query: str) -> str:
        parts: List[str] = []
        parts.append(f"# Codebase Context for Query: '{query}'\n")

        summary = self._generate_context_summary(nodes, relationships)
        parts.append("## Summary")
        parts.append(f"- Found {summary['total_nodes']} nodes across {summary['coverage']['files']} files")
        parts.append(f"- {summary['coverage']['components']} components involved")
        parts.append(f"- {summary['total_relationships']} relationships\n")

        by_file: Dict[str, List[KGNode]] = {}
        for n in nodes:
            by_file.setdefault(n.filePath or "unknown", []).append(n)

        for fp, fnodes in by_file.items():
            if fp == "unknown":
                continue
            parts.append(f"## File: {fp}")

            comps = [n for n in fnodes if n.type == "Component"]
            hooks = [n for n in fnodes if n.type == "Hook"]
            states = [n for n in fnodes if n.type == "State"]
            props = [n for n in fnodes if n.type == "Prop"]
            jsx = [n for n in fnodes if n.type == "JSXElement"]
            others = [n for n in fnodes if n.type not in {"Component","Hook","State","Prop","JSXElement"}]

            if comps:
                parts.append("### Components:")
                for c in comps:
                    parts.append(f"- **{c.name}** ({c.exportType or 'unknown export'})")
                    if c.props: parts.append(f"  - Props: {c.props}")
                    if c.hooksUsed: parts.append(f"  - Hooks: {c.hooksUsed}")
            if hooks:
                parts.append("### Hooks:"); [parts.append(f"- {h.name}") for h in hooks]
            if states:
                parts.append("### State:"); [parts.append(f"- {s.name}") for s in states]
            if jsx:
                parts.append("### JSX Elements:")
                counts: Dict[str, int] = {}
                for j in jsx:
                    tag = j.tagName or "unknown"
                    counts[tag] = counts.get(tag, 0) + 1
                for tag, cnt in counts.items():
                    parts.append(f"- {tag}: {cnt} instances")
            if others:
                parts.append("### Other:")
                for o in others:
                    parts.append(f"- {o.type}: {o.name or o.id}")

            parts.append("")

        if relationships:
            parts.append("## Key Relationships:")
            agg: Dict[str, int] = {}
            for r in relationships:
                agg[r.type] = agg.get(r.type, 0) + 1
            for t, c in sorted(agg.items(), key=lambda x: x[1], reverse=True):
                parts.append(f"- {t}: {c} connections")

        full = "\n".join(parts)
        if len(full) > self.config.max_context_length:
            cut = full[:self.config.max_context_length]
            i = cut.rfind("\n")
            if i > 0:
                full = cut[:i] + "\n\n[Context truncated for length]"
        return full

    def _compose_ollama_prompt(
        self,
        question: str,
        nodes: List[KGNode],
        relationships: List[KGRelationship],
        flow_focus: Optional[Tuple[str, str]] = None,
        flow_paths: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        bundle = self._build_graph_bundle(nodes, relationships)
        bundle_json = json.dumps(bundle, indent=2)[:12000]
        context_text = self._format_for_llm(nodes, relationships, question)
        parts: List[str] = []
        parts.append("You are a senior UI engineer. You are given a knowledge graph extracted from a large React UI codebase.")
        parts.append("Answer strictly based on the provided graph. Prefer Components and their relationships (props, hooks, JSX, routes, imports).")
        parts.append("Always reference component names and file paths when making claims. Avoid generic observations (e.g., Tailwind) unless directly relevant.")
        if flow_focus:
            a, b = flow_focus
            parts.append(f"Focus: Summarize the flow from '{a}' to '{b}'.")
        parts.append("If multiple candidates exist, rank them and explain why. Suggest next files to inspect if needed.")
        parts.append("")
        parts.append(f"Question:\n{question}")
        parts.append("")
        if flow_paths:
            parts.append("Flow Candidates (paths):")
            for i, p in enumerate(flow_paths, 1):
                seq = []
                for n in p.get("nodes", []):
                    label = f"{n.get('type', '')}:{n.get('name') or n.get('id')}"
                    fp = n.get('filePath') or ''
                    if fp:
                        label += f" [{fp}]"
                    seq.append(label)
                parts.append(f"- Path {i}: " + " -> ".join(seq))
            parts.append("")
        parts.append("Graph Summary:")
        parts.append(context_text)
        parts.append("")
        parts.append("Graph Bundle (JSON):")
        parts.append(bundle_json)
        parts.append("")
        parts.append("Output format:")
        parts.append("- Key components and files involved")
        if flow_focus:
            parts.append("- Step-by-step flow from source to target (with file paths)")
        parts.append("- Relevant relationships that enable the flow")
        parts.append("- Gaps/uncertainties and next files to check")
        return "\n".join(parts)

    # ---------------- Utilities ---------------------

    def _unique_nodes(self, nodes: List[KGNode]) -> List[KGNode]:
        seen = set(); out: List[KGNode] = []
        for n in nodes:
            if n.id not in seen:
                out.append(n); seen.add(n.id)
        return out

    def _record_to_node(self, record: Dict[str, Any]) -> KGNode:
        d = dict(record)

        # Normalize common extended fields
        attrs = d.get("attributes", {}) or {}
        if isinstance(attrs, str):
            try:
                parsed = json.loads(attrs)
                attrs = parsed if isinstance(parsed, dict) else {}
            except Exception:
                attrs = {}
        return KGNode(
            id=d.get("id", ""),
            type=d.get("type", ""),
            name=d.get("name"),
            filePath=d.get("filePath"),
            component=d.get("component"),
            exportType=d.get("exportType"),
            domId=d.get("domId"),
            componentId=d.get("componentId"),
            expression=d.get("expression"),
            tagName=d.get("tagName"),
            attributes=attrs,
            classNames=d.get("classNames", []) or [],
            props=d.get("props", []) or [],
            hooksUsed=d.get("hooksUsed", []) or [],
            contextUsed=d.get("contextUsed", []) or [],
            properties={k: v for k, v in d.items() if k not in {
                "id","type","name","filePath","component","exportType",
                "domId","componentId","expression","tagName","attributes",
                "classNames","props","hooksUsed","contextUsed"
            }}
        )

    def _record_to_relationship(self, record: Dict[str, Any]) -> KGRelationship:
        return KGRelationship(
            from_node=record.get("from", ""),
            to_node=record.get("to", ""),
            type=record.get("type", ""),
            intraFile=record.get("intraFile"),
            fileId=record.get("fileId"),
            properties={k: v for k, v in record.items() if k not in {"from","to","type","intraFile","fileId"}}
        )

    def _generate_context_summary(self, nodes: List[KGNode], relationships: List[KGRelationship]) -> Dict[str, Any]:
        node_types: Dict[str, int] = {}
        files, components = set(), set()
        for n in nodes:
            node_types[n.type] = node_types.get(n.type, 0) + 1
            if n.filePath: files.add(n.filePath)
            if n.component: components.add(n.component)
        rel_types: Dict[str, int] = {}
        for r in relationships:
            rel_types[r.type] = rel_types.get(r.type, 0) + 1
        return {
            "total_nodes": len(nodes),
            "total_relationships": len(relationships),
            "node_types": node_types,
            "relationship_types": rel_types,
            "files_involved": list(files)[:10],
            "components_involved": list(components)[:10],
            "coverage": {"files": len(files), "components": len(components)}
        }

    # ---------------- Public API --------------------

    def get_component_context(self, component_name: str, expand_depth: int = 1) -> Tuple[List[KGNode], List[KGRelationship]]:
        cypher = """
        MATCH (c:KGNode {type: 'Component'})
        WHERE c.name = $component_name OR c.name CONTAINS $component_name
        RETURN c
        LIMIT 5
        """
        rows = self._execute_query(cypher, {"component_name": component_name})
        if not rows: return [], []
        component_ids = [row["c"]["id"] for row in rows]
        components = [self._record_to_node(row["c"]) for row in rows]
        expanded_nodes, rels = self.expand_context(component_ids, expand_depth)
        all_nodes = self._unique_nodes(components + expanded_nodes)
        return all_nodes, rels

    def search(self, request: ContextRequest) -> ContextResponse:
        nodes: List[KGNode] = []
        relationships: List[KGRelationship] = []

        if request.search_type == SearchType.FULLTEXT:
            nodes = self.fulltext_search(request.query, request.max_results)
        elif request.search_type == SearchType.COMPONENT:
            nodes, relationships = self.get_component_context(request.query, request.expand_depth)
        elif request.search_type == SearchType.FILE:
            nodes = self.file_search(request.query, request.max_results)
        elif request.search_type == SearchType.SEMANTIC:
            nodes, relationships = self.semantic_component_retrieval(
                query=request.query, k=request.max_results, expand_depth=request.expand_depth
            )

        if request.file_filter:
            nodes = [n for n in nodes if request.file_filter.lower() in (n.filePath or "").lower()]
        if request.component_filter:
            nodes = [n for n in nodes if request.component_filter.lower() in (n.component or "").lower()]

        if request.expand_depth > 0 and request.search_type not in [SearchType.COMPONENT]:
            ids = [n.id for n in nodes]
            extra_nodes, extra_rels = self.expand_context(ids, request.expand_depth)
            nodes = self._unique_nodes(nodes + extra_nodes)
            relationships.extend(extra_rels)
            nodes = nodes[:request.max_results * 2]

        context_summary = self._generate_context_summary(nodes, relationships)
        llm_context = self._format_for_llm(nodes, relationships, request.query)

        return ContextResponse(
            query=request.query,
            search_type=request.search_type.value,
            results=nodes,
            relationships=relationships,
            context_summary=context_summary,
            llm_formatted_context=llm_context
        )

    def ask(self, req: AskRequest) -> AskResponse:
        # 1) Planner (contextually-aware)
        plan = self._planner(req.question)
        # Safe coercion for expand_depth from planner or request
        plan_depth = plan.get("expand_depth", req.expand_depth)
        try:
            expand_depth = int(plan_depth)
        except (ValueError, TypeError):
            expand_depth = req.expand_depth
        expand_depth = max(0, min(3, expand_depth))

        # Optional flow extraction (e.g., from Login to Analytics)
        flow_pair = self._extract_flow_pair(req.question, plan)

        # 2) Forced paths if provided
        if req.strategy == "component" and req.component_name:
            nodes, rels = self.get_component_context(req.component_name, expand_depth)
        elif req.strategy == "file" and req.file_path:
            nodes = self.file_search(req.file_path, req.max_results)
            ids = [n.id for n in nodes]
            extra_nodes, rels = self.expand_context(ids, expand_depth)
            nodes = self._unique_nodes(nodes + extra_nodes)
        else:
            # 3) Hybrid pool driven by planner
            cand = self._hybrid_candidates(plan, req.question, k_sem=max(20, req.max_results))

            # If flow is detected, explicitly include source/target component contexts in candidates
            if flow_pair:
                src_name, dst_name = flow_pair
                src_nodes_ctx, _r1 = self.get_component_context(src_name, expand_depth=1)
                dst_nodes_ctx, _r2 = self.get_component_context(dst_name, expand_depth=1)
                pool = {n.id: n for n in cand}
                for n in (src_nodes_ctx + dst_nodes_ctx):
                    pool[n.id] = n
                cand = list(pool.values())

            seed_ids = [n.id for n in cand if n.type == "Component"][:8] or [n.id for n in cand[:5]]

            # If flow is detected, make sure source/target components are part of seeds
            if flow_pair:
                src_name, dst_name = flow_pair
                src_ids = self._find_component_ids_by_name(src_name)
                dst_ids = self._find_component_ids_by_name(dst_name)
                seed_ids = list(dict.fromkeys(src_ids + dst_ids + seed_ids))[:10]

            expanded_nodes, rels = self.expand_context(seed_ids, expand_depth)
            nodes = self._unique_nodes(cand + expanded_nodes)

            # 4) Final graph-aware re-rank (cross-encoder if available)
            nodes = self._rerank_nodes_graphaware(
                query=req.question, nodes=nodes, seed_ids=seed_ids, rels=rels, top_n=req.max_results
            )

            # Ensure source/target components are present in final top-N if flow is requested
            if flow_pair:
                src_name, dst_name = flow_pair
                have_ids = {n.id for n in nodes}
                add_nodes: List[KGNode] = []
                for name in (src_name, dst_name):
                    ctx_nodes, _r = self.get_component_context(name, expand_depth=0)
                    comp_nodes = [n for n in ctx_nodes if n.type == "Component"]
                    if comp_nodes:
                        c0 = comp_nodes[0]
                        if c0.id not in have_ids:
                            add_nodes.append(c0)
                if add_nodes:
                    nodes = self._unique_nodes(add_nodes + nodes)[:req.max_results]

        flow_paths: Optional[List[Dict[str, Any]]] = None
        if flow_pair:
            src_name, dst_name = flow_pair
            src_ids = self._find_component_ids_by_name(src_name)
            dst_ids = self._find_component_ids_by_name(dst_name)
            if src_ids and dst_ids:
                flow_paths = self._find_paths_between_components(src_ids, dst_ids, max_len=4, limit=5)

        prompt = self._compose_ollama_prompt(req.question, nodes, rels, flow_focus=flow_pair, flow_paths=flow_paths)
        answer = self._call_ollama(prompt, model=req.model)

        retrieval_info = {
            "total_nodes": len(nodes),
            "total_relationships": len(rels),
            "top_files": list({n.filePath for n in nodes if n.filePath})[:10],
            "top_components": [n.name for n in nodes if n.type == "Component"][:10],
            "planner": plan,
            "flow_focus": {"from": flow_pair[0], "to": flow_pair[1]} if flow_pair else None,
            "flow_paths_count": len(flow_paths) if flow_paths else 0,
        }
        return AskResponse(question=req.question, answer=answer, retrieval=retrieval_info)

# ------------------------- FastAPI -------------------------------

config = RetrievalConfig()
retriever = KnowledgeGraphRetriever(config)

app = FastAPI(
    title="Knowledge Graph Retrieval Service",
    description="Intelligent retrieval system for UI codebase knowledge graphs (Upgraded)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("shutdown")
def shutdown_event():
    retriever.close()

@app.get("/")
def read_root():
    return {"message": "Knowledge Graph Retrieval Service", "version": "2.0.0"}

@app.get("/health")
def health_check():
    index_ready = isinstance(retriever._model, SentenceTransformer) and len(retriever._component_index) > 0
    cross_ready = bool(retriever._cross) if retriever.config.use_cross_encoder else False
    return {
        "status": "healthy",
        "neo4j_connected": True,
        "embedding_index_ready": index_ready,
        "cross_encoder_ready": cross_ready
    }

@app.post("/search", response_model=ContextResponse)
def search_context(request: ContextRequest):
    try:
        return retriever.search(request)
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/component/{component_name}")
def search_component(
    component_name: str,
    expand_depth: int = Query(1, ge=0, le=3),
    max_results: int = Query(20, ge=1, le=100)
):
    request = ContextRequest(
        query=component_name, search_type=SearchType.COMPONENT,
        expand_depth=expand_depth, max_results=max_results
    )
    return retriever.search(request)

@app.get("/search/file")
def search_file(
    file_path: str = Query(..., description="File path to search"),
    max_results: int = Query(50, ge=1, le=200)
):
    request = ContextRequest(query=file_path, search_type=SearchType.FILE, max_results=max_results)
    return retriever.search(request)

@app.get("/search/fulltext")
def search_fulltext(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=50),
    expand_depth: int = Query(0, ge=0, le=2)
):
    request = ContextRequest(
        query=q, search_type=SearchType.FULLTEXT,
        max_results=max_results, expand_depth=expand_depth
    )
    return retriever.search(request)

@app.get("/search/semantic")
def search_semantic(
    q: str = Query(..., description="Search query (components only)"),
    max_results: int = Query(10, ge=1, le=50),
    expand_depth: int = Query(1, ge=0, le=3)
):
    request = ContextRequest(
        query=q, search_type=SearchType.SEMANTIC,
        max_results=max_results, expand_depth=expand_depth
    )
    return retriever.search(request)

@app.get("/stats")
def get_stats():
    try:
        node_stats = retriever._execute_query("""
            MATCH (n:KGNode)
            RETURN n.type AS node_type, count(*) AS count
            ORDER BY count DESC
        """)
        rel_stats = retriever._execute_query("""
            MATCH ()-[r:KGR]->()
            RETURN r.type AS rel_type, count(*) AS count
            ORDER BY count DESC
        """)
        return {"node_statistics": node_stats, "relationship_statistics": rel_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask_codebase(req: AskRequest):
    try:
        return retriever.ask(req)
    except Exception as e:
        logger.error("/ask failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
