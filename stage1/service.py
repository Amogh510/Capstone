"""
Stage 1 — Scenario Interpretation + Similarity Search

Implements:
- Stage 1A: KG Entity Extraction (Routes + Components) via semantic similarity
- Stage 1B: Scenario Interpretation (LLM-assisted, with deterministic fallback)

Neo4j vector search is preferred when vector indexes exist.
Fallback uses local cosine similarity over cached node embeddings.

Environment variables:
- NEO4J_URI, NEO4J_USER, NEO4J_PASS (or NEO4J_PASSWORD)
- STAGE1_EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
- STAGE1_COMPONENT_INDEX (default: component_embeddings)
- STAGE1_ROUTE_INDEX (default: route_embeddings)
- LITELLM_MODEL (optional; e.g., gpt-4o-mini, openai/gpt-4o-mini, etc.)

Usage:
>>> from stage1.service import Stage1Service
>>> svc = Stage1Service()
>>> out = svc.run_stage1("Login → Dashboard → Todos CRUD")
>>> print(out)
"""

from __future__ import annotations

import os
import json
import math
import time
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import sys

import numpy as np
from neo4j import GraphDatabase

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not required

# Embeddings via LangChain (HuggingFace local by default)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    HuggingFaceEmbeddings = None  # type: ignore

# Optional LLM (LiteLLM via LangChain). We keep it best-effort.
try:
    from langchain_community.chat_models import ChatLiteLLM  # type: ignore
    from langchain_core.messages import HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    ChatLiteLLM = None  # type: ignore
    HumanMessage = None  # type: ignore


def _to_float_list(val: Any) -> Optional[List[float]]:
    if val is None:
        return None
    if isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
        return [float(x) for x in val]
    # Some drivers can return numpy arrays or strings; try to coerce
    try:
        if hasattr(val, 'tolist'):
            return [float(x) for x in val.tolist()]
        if isinstance(val, str):
            arr = json.loads(val)
            if isinstance(arr, list):
                return [float(x) for x in arr]
    except Exception:
        return None
    return None


def _cosine_similarity(vec_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a (d,) and B (n,d) safely."""
    a = vec_a / (np.linalg.norm(vec_a) + 1e-12)
    B = mat_b / (np.linalg.norm(mat_b, axis=1, keepdims=True) + 1e-12)
    return B @ a


@dataclass
class Stage1Config:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass: str = os.getenv("NEO4J_PASS", os.getenv("NEO4J_PASSWORD", "admin123"))

    component_index: str = os.getenv("STAGE1_COMPONENT_INDEX", "component_embeddings")
    route_index: str = os.getenv("STAGE1_ROUTE_INDEX", "route_embeddings")

    embedding_model: str = os.getenv("STAGE1_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cache_dir: str = os.getenv("STAGE1_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache", "stage1"))

    # Top-k per type
    top_k_components: int = int(os.getenv("STAGE1_TOPK_COMPONENTS", "5"))
    top_k_routes: int = int(os.getenv("STAGE1_TOPK_ROUTES", "5"))

    # LLM for Stage 1B
    litellm_model: Optional[str] = os.getenv("LITELLM_MODEL")
    llm_temperature: float = float(os.getenv("STAGE1_LLM_TEMPERATURE", "0.0"))
    llm_max_tokens: int = int(os.getenv("STAGE1_LLM_MAX_TOKENS", "500"))
    enable_stage1b: bool = os.getenv("STAGE1_ENABLE_STAGE1B", "true").lower() in ("true", "1", "yes")


class Stage1Service:
    def __init__(self, config: Optional[Stage1Config] = None):
        self.config = config or Stage1Config()
        os.makedirs(self.config.cache_dir, exist_ok=True)
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_pass),
        )
        # Lazy embeddings
        self._emb = None

    # --------------- Core public API ---------------
    def run_stage1(self, scenario: str) -> Dict[str, Any]:
        """Run Stage 1 (A+B) and return routes/components with scores.

        Returns structure:
        {
          "scenario": <input>,
          "interpretation": {"steps": [...], "keywords": [...]},
          "routes": [{"name": str, "score": float}],
          "components": [{"name": str, "score": float}],
          "structured": {  # Stage 1B output (if enabled)
            "type": "workflow" | "page" | "component",
            "routes": [str],
            "components": [str],
            "auth_required": bool,
            "goal": str,
            "priority": "high" | "medium" | "low"
          }
        }
        """
        # Stage 1A: Entity Extraction via similarity search
        interpretation = self._interpret_scenario(scenario)
        query_text = self._compose_query_text(scenario, interpretation)
        q_emb = self._embed_text(query_text)

        # Pass steps for position-aware route boosting
        steps = interpretation.get("steps", [])
        routes = self._search_routes(q_emb, steps=steps)
        components = self._search_components(q_emb)

        result = {
            "scenario": scenario,
            "interpretation": interpretation,
            "routes": routes,
            "components": components,
        }

        # Stage 1B: LLM-based structured interpretation
        if self.config.enable_stage1b:
            structured = self._interpret_scenario_with_llm(scenario, routes, components)
            if structured:
                result["structured"] = structured

        return result

    # --------------- Scenario Interpretation ---------------
    def _interpret_scenario(self, scenario: str) -> Dict[str, Any]:
        """LLM-assisted interpretation with safe fallback.
        Output keys: steps: List[str], keywords: List[str]
        """
        # Try LiteLLM via LangChain if configured
        if self.config.litellm_model and ChatLiteLLM and HumanMessage:
            try:
                llm = ChatLiteLLM(model=self.config.litellm_model, temperature=0.0)
                prompt = (
                    "You are a test scenario interpreter. Given a UI test scenario, "
                    "extract two arrays: 'steps' (ordered, concise step names) and 'keywords' (core nouns/verbs). "
                    "Return ONLY valid minified JSON with keys steps and keywords.\n\n"
                    f"Scenario: {scenario}\nJSON:"
                )
                resp = llm.invoke([HumanMessage(content=prompt)])
                content = getattr(resp, "content", "")
                # Extract JSON object
                s = content.find("{")
                e = content.rfind("}")
                if s != -1 and e != -1:
                    out = json.loads(content[s : e + 1])
                    steps = out.get("steps") or []
                    keywords = out.get("keywords") or []
                    return {
                        "steps": [str(x).strip() for x in steps if str(x).strip()],
                        "keywords": [str(x).strip() for x in keywords if str(x).strip()],
                    }
            except Exception:
                pass

        # Deterministic fallback: split on arrows, commas, then words
        raw = scenario.replace("→", "->").replace("—", "->")
        parts = [p.strip() for p in raw.replace("->", ">").split(">") if p.strip()]
        steps = parts if parts else [scenario.strip()]
        # Extract basic keywords: alphanumerics from steps
        import re as _re
        kw = []
        for step in steps:
            toks = _re.findall(r"[A-Za-z][A-Za-z0-9_\-/]+", step)
            for t in toks:
                tt = t.strip().lower()
                if tt and tt not in kw:
                    kw.append(tt)
        return {"steps": steps, "keywords": kw[:12]}

    def _compose_query_text(self, scenario: str, interp: Dict[str, Any]) -> str:
        steps = interp.get("steps") or []
        keywords = interp.get("keywords") or []
        # A compact enriched query text for embedding
        parts = [scenario]
        if steps:
            parts.append(" | ".join(steps))
        if keywords:
            parts.append(", ".join(keywords))
        return "\n".join(parts)

    # --------------- Stage 1B: LLM-based Structured Interpretation ---------------
    def _interpret_scenario_with_llm(
        self, 
        scenario: str, 
        routes: List[Dict[str, Any]], 
        components: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Stage 1B: Use LLM to produce structured interpretation using ONLY KG entities.
        
        Output schema:
        {
          "type": "workflow" | "page" | "component",
          "routes": [str],
          "components": [str],
          "auth_required": bool,
          "goal": str,
          "priority": "high" | "medium" | "low"
        }
        """
        if not self.config.litellm_model or not ChatLiteLLM or not HumanMessage:
            return None

        try:
            llm = ChatLiteLLM(
                model=self.config.litellm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )

            # Build the prompt
            system_prompt = """You are a Scenario Interpreter for the TestLab React app.
Use ONLY the entities provided below from the knowledge graph.
Do not invent or assume any new routes or components.
Output strictly valid JSON matching the schema.

Output JSON Schema:
{
  "type": "workflow" | "page" | "component",
  "routes": [string],  // Select relevant routes from provided list
  "components": [string],  // Select relevant components from provided list
  "auth_required": boolean,  // Does this scenario require authentication?
  "goal": string,  // Brief description of what the scenario accomplishes
  "priority": "high" | "medium" | "low"  // Based on workflow complexity and criticality
}

Rules:
- If scenario has multiple steps (e.g., Login → Dashboard → Feature), type is "workflow"
- If scenario focuses on a single page, type is "page"
- If scenario focuses on a single UI component interaction, type is "component"
- Select routes in the order they appear in the user flow
- Include only components that are directly involved
- Priority: "high" for auth/critical flows, "medium" for dashboard/settings, "low" for simple views
"""

            # Format routes and components for the prompt
            routes_text = json.dumps([{"name": r["name"], "score": round(r["score"], 3)} for r in routes], indent=2)
            components_text = json.dumps([{"name": c["name"], "score": round(c["score"], 3)} for c in components], indent=2)

            user_prompt = f"""Scenario: {scenario}

AVAILABLE ROUTES (with similarity scores):
{routes_text}

AVAILABLE COMPONENTS (with similarity scores):
{components_text}

Produce the JSON interpretation:"""

            # Invoke LLM
            messages = [
                HumanMessage(content=system_prompt + "\n\n" + user_prompt)
            ]
            resp = llm.invoke(messages)
            content = getattr(resp, "content", "")

            # Extract JSON from response
            s = content.find("{")
            e = content.rfind("}")
            if s != -1 and e != -1:
                result = json.loads(content[s : e + 1])
                
                # Validate schema
                required_keys = {"type", "routes", "components", "auth_required", "goal", "priority"}
                if not required_keys.issubset(result.keys()):
                    return None
                
                # Validate enum values
                if result["type"] not in ["workflow", "page", "component"]:
                    return None
                if result["priority"] not in ["high", "medium", "low"]:
                    return None
                if not isinstance(result["auth_required"], bool):
                    return None
                if not isinstance(result["routes"], list) or not isinstance(result["components"], list):
                    return None
                
                return result
            
            return None

        except Exception as e:
            # Log error for debugging, but don't crash - Stage 1B is optional
            import sys
            print(f"[Stage 1B] LLM interpretation failed: {type(e).__name__}: {e}", file=sys.stderr)
            return None

    # --------------- Embeddings ---------------
    def _ensure_embeddings(self):
        if self._emb is None:
            if HuggingFaceEmbeddings is None:
                raise RuntimeError("langchain_community is not installed; cannot load embeddings.")
            self._emb = HuggingFaceEmbeddings(model_name=self.config.embedding_model, encode_kwargs={"normalize_embeddings": True})

    def _embed_text(self, text: str) -> np.ndarray:
        self._ensure_embeddings()
        vec = self._emb.embed_query(text)
        return np.asarray(vec, dtype=np.float32)

    # --------------- Neo4j helpers ---------------
    def _query_neo4j(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]

    # --------------- Vector search (preferred) ---------------
    def _vector_search(self, index_name: str, q_emb: np.ndarray, top_k: int) -> Optional[List[Tuple[str, float]]]:
        """Return list of (name, score). None if vector index invocation fails."""
        try:
            cypher = (
                "CALL db.index.vector.queryNodes($index, $topK, $embedding) "
                "YIELD node, score RETURN node.name AS name, node.path AS path, score"
            )
            rows = self._query_neo4j(cypher, {"index": index_name, "topK": int(top_k), "embedding": q_emb.tolist()})
            out: List[Tuple[str, float]] = []
            for r in rows:
                name = r.get("name") or r.get("path")  # Use path for Route nodes
                score = r.get("score")
                if name is None or score is None:
                    continue
                # Some Neo4j setups return distance; if so, convert to similarity heuristic
                s = float(score)
                if s > 2.0:  # likely Euclidean distance; compress to (0,1]
                    s = 1.0 / (1.0 + s)
                out.append((str(name), float(s)))
            # Deduplicate by name, keep best score
            best: Dict[str, float] = {}
            for n, s in out:
                if n not in best or s > best[n]:
                    best[n] = s
            return [(n, best[n]) for n in sorted(best, key=lambda x: best[x], reverse=True)][:top_k]
        except Exception:
            return None

    # --------------- Local similarity fallback ---------------
    def _cache_path(self, kind: str) -> str:
        return os.path.join(self.config.cache_dir, f"{kind}.pkl")

    def _load_or_build_cache(self, kind: str) -> Tuple[List[str], np.ndarray]:
        """kind in {components, routes}. Returns (names, embeddings)."""
        pkl = self._cache_path(kind)
        if os.path.exists(pkl):
            try:
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                names: List[str] = data["names"]
                emb: np.ndarray = data["embeddings"]
                if isinstance(emb, list):
                    emb = np.asarray(emb, dtype=np.float32)
                return names, emb
            except Exception:
                pass

        # Build from Neo4j
        type_filter = "Component" if kind == "components" else "Route"
        rows = self._query_neo4j(
            "MATCH (n:KGNode {type: $type}) RETURN n.name AS name, n.path AS path, n.description AS description, n.embedding AS embedding",
            {"type": type_filter},
        )
        names: List[str] = []
        embed_list: List[np.ndarray] = []

        for r in rows:
            name = r.get("name") or r.get("path") or ""  # Use path for Route nodes
            emb = _to_float_list(r.get("embedding"))
            if emb is None:
                # Compute embedding from text
                desc = r.get("description") or ""
                text = f"{type_filter}: {name}. {desc}".strip()
                vec = self._embed_text(text)
            else:
                vec = np.asarray(emb, dtype=np.float32)
            names.append(str(name))
            embed_list.append(vec)

        if not embed_list:
            return [], np.zeros((0, 384), dtype=np.float32)

        mat = np.vstack(embed_list)
        try:
            with open(pkl, "wb") as f:
                pickle.dump({"names": names, "embeddings": mat}, f)
        except Exception:
            pass
        return names, mat

    # --------------- Search orchestrators ---------------
    def _search_components(self, q_emb: np.ndarray) -> List[Dict[str, Any]]:
        # Prefer Neo4j vector index
        topk = self.config.top_k_components
        indexed = self._vector_search(self.config.component_index, q_emb, topk)
        if indexed is not None and len(indexed) > 0:
            return [{"name": n, "score": float(s)} for n, s in indexed]

        # Fallback to local similarity
        names, mat = self._load_or_build_cache("components")
        if mat.shape[0] == 0:
            return []
        sims = _cosine_similarity(q_emb, mat)
        order = np.argsort(-sims)[:topk]
        return [{"name": names[i], "score": float(sims[i])} for i in order]

    def _search_routes(self, q_emb: np.ndarray, steps: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Multi-stage route retrieval with per-step search and position-aware scoring."""
        topk = self.config.top_k_routes
        
        # Stage 1: Global query search (lower weight)
        global_results = self._search_routes_single(q_emb, topk * 3)
        route_scores: Dict[str, float] = {}
        for r in global_results:
            route_scores[r["name"]] = r["score"] * 0.4  # 40% weight for global query
        
        # Stage 2: Per-step search with position-aware weights (higher weight)
        if steps:
            step_weights = self._compute_step_weights(len(steps))
            for i, step in enumerate(steps):
                step_emb = self._embed_text(step)
                step_results = self._search_routes_single(step_emb, topk * 2)
                
                # Merge with position-aware scoring
                for r in step_results:
                    route_name = r["name"]
                    # Step-specific score weighted by position (60% total weight distributed)
                    contribution = r["score"] * step_weights[i]
                    if route_name in route_scores:
                        route_scores[route_name] += contribution
                    else:
                        route_scores[route_name] = contribution
        
        # Stage 3: Keyword boosting (additional bonus for exact matches)
        if steps:
            for route_name in route_scores:
                boost = self._compute_keyword_boost(route_name, steps)
                route_scores[route_name] += boost
        
        # Sort and return top-k
        sorted_routes = sorted(
            [{"name": name, "score": score} for name, score in route_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        return sorted_routes[:topk]
    
    def _compute_step_weights(self, num_steps: int) -> List[float]:
        """Compute position-aware weights for steps (earlier = more important)."""
        if num_steps == 0:
            return []
        if num_steps == 1:
            return [0.6]
        if num_steps == 2:
            return [0.35, 0.25]  # First step gets more weight
        if num_steps == 3:
            return [0.30, 0.20, 0.10]
        # For 4+ steps, distribute with decay
        weights = []
        remaining = 0.6
        for i in range(num_steps):
            weight = remaining * (0.6 ** i) / sum(0.6 ** j for j in range(num_steps))
            weights.append(weight)
        return weights
    
    def _compute_keyword_boost(self, route_name: str, steps: List[str]) -> float:
        """Additional boost for exact keyword matches in route path."""
        route_lower = route_name.lower()
        boost = 0.0
        for i, step in enumerate(steps):
            keywords = [w.lower() for w in step.split() if len(w) > 2]
            for keyword in keywords:
                if keyword in route_lower:
                    # Higher boost for earlier steps, diminishing for later
                    position_boost = 0.12 / (i + 1)
                    boost += position_boost
                    break  # Only count once per step
        return boost
    
    def _search_routes_single(self, q_emb: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Single query route search (used by multi-stage search)."""
        # Try Neo4j vector index first
        indexed = self._vector_search(self.config.route_index, q_emb, limit)
        if indexed is not None and len(indexed) > 0:
            return [{"name": n, "score": float(s)} for n, s in indexed]
        
        # Fallback: local embeddings
        names, mat = self._load_or_build_cache("routes")
        if mat.shape[0] == 0:
            return []
        sims = _cosine_similarity(q_emb, mat)
        order = np.argsort(-sims)[:limit]
        return [{"name": names[i], "score": float(sims[i])} for i in order]


# --------------- CLI ---------------
def _main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 — Scenario Interpretation + Similarity Search")
    # scenario can be provided positionally or via --scenario; both are optional with fallbacks
    parser.add_argument("scenario", nargs="?", default=None, type=str, help="Test scenario text, e.g. 'Login → Dashboard → Todos CRUD'")
    parser.add_argument("-s", "--scenario", dest="scenario_flag", type=str, default=None, help="Scenario text (alternative to positional)")
    parser.add_argument("--topk-components", type=int, default=None, help="Top-k components to return")
    parser.add_argument("--topk-routes", type=int, default=None, help="Top-k routes to return")
    parser.add_argument("--disable-stage1b", action="store_true", help="Disable Stage 1B LLM interpretation")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model for Stage 1B (e.g., gpt-4o-mini)")
    args = parser.parse_args()

    cfg = Stage1Config()
    if args.topk_components is not None:
        cfg.top_k_components = args.topk_components
    if args.topk_routes is not None:
        cfg.top_k_routes = args.topk_routes
    if args.disable_stage1b:
        cfg.enable_stage1b = False
    if args.llm_model is not None:
        cfg.litellm_model = args.llm_model

    # Resolve scenario: flag > positional > env > stdin (if piped)
    scenario = args.scenario_flag or args.scenario or os.getenv("STAGE1_SCENARIO")
    if not scenario:
        try:
            if not sys.stdin.isatty():
                data = sys.stdin.read().strip()
                if data:
                    scenario = data
        except Exception:
            pass

    if not scenario or not str(scenario).strip():
        parser.print_usage(sys.stderr)
        print("error: scenario is required (provide positionally, --scenario, STAGE1_SCENARIO, or via stdin)", file=sys.stderr)
        sys.exit(2)

    svc = Stage1Service(cfg)
    out = svc.run_stage1(str(scenario))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _main_cli()


