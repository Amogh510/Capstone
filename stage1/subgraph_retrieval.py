"""
Stage 2A — Subgraph Retrieval (Context Extraction from KG)

Retrieves relevant subgraphs from Neo4j Knowledge Graph based on
routes and components identified in Stage 1.

The subgraph includes:
- Specified routes and components
- All connected entities (States, EventHandlers, Props, Hooks, JSXElements, etc.)
- Configurable depth traversal (default: 2)

Environment variables:
- NEO4J_URI, NEO4J_USER, NEO4J_PASS (or NEO4J_PASSWORD)
- STAGE2_MAX_DEPTH (default: 2)
- STAGE2_INCLUDE_FILE_CONTEXT (default: true)

Usage:
>>> from stage1.subgraph_retrieval import retrieve_subgraph
>>> structured_input = {
...     "routes": ["/dashboard", "/dashboard/analytics"],
...     "components": ["DashboardLayout", "Analytics"]
... }
>>> subgraph = retrieve_subgraph(structured_input, depth=2)
>>> print(subgraph)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import GraphDatabase

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass


@dataclass
class Stage2Config:
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass: str = os.getenv("NEO4J_PASS", os.getenv("NEO4J_PASSWORD", "admin123"))
    
    max_depth: int = int(os.getenv("STAGE2_MAX_DEPTH", "2"))
    include_file_context: bool = os.getenv("STAGE2_INCLUDE_FILE_CONTEXT", "false").lower() in ("true", "1", "yes")
    
    # Smart filtering: exclude low-value node types for test generation
    exclude_node_types: List[str] = None
    include_only_node_types: List[str] = None  # If set, only include these types
    
    # Aggregation: summarize instead of individual nodes
    aggregate_styling: bool = os.getenv("STAGE2_AGGREGATE_STYLING", "true").lower() in ("true", "1", "yes")
    max_jsx_depth: int = int(os.getenv("STAGE2_MAX_JSX_DEPTH", "1"))  # Limit JSX tree depth
    
    # Relationship types to traverse (ordered by priority)
    relationship_types: List[str] = None
    
    def __post_init__(self):
        if self.relationship_types is None:
            self.relationship_types = [
                "componentDeclaresState",
                "componentDefinesEventHandler",
                "componentHasProp",
                "componentUsesHook",
                "componentUsesContext",
                "componentRendersJsx",
                "routeRendersComponent",
                "componentRendersComponent",
                "fileContainsKgNode",
            ]
        
        # Default: exclude styling noise for test generation
        if self.exclude_node_types is None:
            self.exclude_node_types = [
                "TailwindUtility",
                "InlineStyle",
            ] if self.aggregate_styling else []


class SubgraphRetriever:
    def __init__(self, config: Optional[Stage2Config] = None):
        self.config = config or Stage2Config()
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_pass),
        )
    
    def close(self):
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # --------------- Core API ---------------
    def retrieve_subgraph(
        self,
        structured_input: Dict[str, Any],
        depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retrieve subgraph from Neo4j containing all relevant context.
        
        Args:
            structured_input: Dict with 'routes' and 'components' lists
            depth: Max traversal depth (default: from config)
        
        Returns:
            {
                "nodes": [{"id": str, "label": str, "properties": dict}],
                "edges": [{"source": str, "target": str, "type": str, "properties": dict}],
                "summary": {"node_count": int, "edge_count": int, "node_types": dict}
            }
        """
        depth = depth if depth is not None else self.config.max_depth
        routes = structured_input.get("routes", [])
        components = structured_input.get("components", [])
        
        if not routes and not components:
            return {
                "nodes": [], 
                "edges": [], 
                "summary": {
                    "node_count": 0,
                    "edge_count": 0,
                    "node_types": {},
                    "edge_types": {}
                }
            }
        
        # Step 1: Find seed nodes (routes and components)
        seed_nodes = self._find_seed_nodes(routes, components)
        
        if not seed_nodes:
            return {
                "nodes": [], 
                "edges": [], 
                "summary": {
                    "node_count": 0,
                    "edge_count": 0,
                    "node_types": {},
                    "edge_types": {}
                }
            }
        
        # Step 2: Expand subgraph from seed nodes
        nodes, edges = self._expand_subgraph(seed_nodes, depth)
        
        # Step 3: Smart filtering to reduce noise
        nodes, edges = self._apply_smart_filtering(nodes, edges)
        
        # Step 4: Optionally include file context
        if self.config.include_file_context:
            nodes, edges = self._add_file_context(nodes, edges)
        
        # Step 5: Aggregate styling information if enabled
        styling_summary = None
        if self.config.aggregate_styling:
            nodes, edges, styling_summary = self._aggregate_styling_nodes(nodes, edges)
        
        # Step 6: Build summary statistics
        summary = self._build_summary(nodes, edges, styling_summary)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "summary": summary
        }
    
    # --------------- Neo4j Queries ---------------
    def _query_neo4j(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]
    
    def _find_seed_nodes(self, routes: List[str], components: List[str]) -> List[Dict[str, Any]]:
        """Find initial nodes matching routes and components."""
        seed_nodes = []
        
        # Find routes
        if routes:
            # Routes can be stored with 'path' or 'name' property
            route_query = """
                MATCH (n:KGNode)
                WHERE n.type = 'Route' AND (n.path IN $routes OR n.name IN $routes)
                RETURN n.id AS id, n.type AS type, n.name AS name, n.path AS path, properties(n) AS props
            """
            route_results = self._query_neo4j(route_query, {"routes": routes})
            seed_nodes.extend(route_results)
        
        # Find components
        if components:
            component_query = """
                MATCH (n:KGNode)
                WHERE n.type = 'Component' AND n.name IN $components
                RETURN n.id AS id, n.type AS type, n.name AS name, properties(n) AS props
            """
            component_results = self._query_neo4j(component_query, {"components": components})
            seed_nodes.extend(component_results)
        
        return seed_nodes
    
    def _expand_subgraph(
        self,
        seed_nodes: List[Dict[str, Any]],
        depth: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Expand subgraph from seed nodes up to specified depth."""
        seed_ids = [node["id"] for node in seed_nodes]
        
        # Query 1: Get all reachable nodes
        node_query = f"""
            MATCH (seed:KGNode)
            WHERE seed.id IN $seed_ids
            OPTIONAL MATCH path = (seed)-[r:KGR*1..{depth}]-(connected:KGNode)
            WITH collect(DISTINCT seed) + collect(DISTINCT connected) AS allNodes
            UNWIND allNodes AS n
            WITH n WHERE n IS NOT NULL
            RETURN DISTINCT n.id AS id, n.type AS type, n.name AS name, n.path AS path, properties(n) AS props
        """
        
        node_results = self._query_neo4j(node_query, {"seed_ids": seed_ids})
        nodes = [self._format_node(n) for n in node_results]
        
        if not nodes:
            return [self._format_node(n) for n in seed_nodes], []
        
        # Query 2: Get all edges between the retrieved nodes
        node_ids = [n["id"] for n in nodes]
        edge_query = """
            MATCH (from:KGNode)-[r:KGR]->(to:KGNode)
            WHERE from.id IN $node_ids AND to.id IN $node_ids
            RETURN DISTINCT from.id AS source, to.id AS target, r.type AS type, properties(r) AS props
        """
        
        edge_results = self._query_neo4j(edge_query, {"node_ids": node_ids})
        edges = []
        for e in edge_results:
            edges.append({
                "source": e["source"],
                "target": e["target"],
                "type": e["type"],
                "properties": e.get("props", {})
            })
        
        return nodes, edges
    
    def _add_file_context(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Add file nodes that contain the KG nodes."""
        node_ids = [n["id"] for n in nodes]
        
        file_query = """
            MATCH (kgNode:KGNode)-[:KGR {type: 'fileContainsKgNode'}]-(file:KGNode {type: 'File'})
            WHERE kgNode.id IN $node_ids
            RETURN DISTINCT file.id AS id, file.type AS type, file.name AS name, properties(file) AS properties
        """
        
        file_nodes = self._query_neo4j(file_query, {"node_ids": node_ids})
        
        if file_nodes:
            # Add file nodes
            file_node_formatted = [self._format_node(f) for f in file_nodes]
            nodes.extend(file_node_formatted)
            
            # Add file containment edges
            edge_query = """
                MATCH (file:KGNode {type: 'File'})-[r:KGR {type: 'fileContainsKgNode'}]->(kgNode:KGNode)
                WHERE kgNode.id IN $node_ids
                RETURN file.id AS source, kgNode.id AS target, 'fileContainsKgNode' AS type, properties(r) AS properties
            """
            file_edges = self._query_neo4j(edge_query, {"node_ids": node_ids})
            edges.extend(file_edges)
        
        return nodes, edges
    
    # --------------- Smart Filtering ---------------
    def _apply_smart_filtering(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply intelligent filtering to reduce noise while keeping valuable context."""
        
        # Filter nodes by type
        if self.config.include_only_node_types:
            # Whitelist mode: only include specified types
            allowed_types = set(self.config.include_only_node_types)
            filtered_nodes = [n for n in nodes if n["label"] in allowed_types]
        elif self.config.exclude_node_types:
            # Blacklist mode: exclude specified types
            excluded_types = set(self.config.exclude_node_types)
            filtered_nodes = [n for n in nodes if n["label"] not in excluded_types]
        else:
            filtered_nodes = nodes
        
        # Keep track of valid node IDs
        valid_node_ids = {n["id"] for n in filtered_nodes}
        
        # Filter edges: only keep edges between valid nodes
        filtered_edges = [
            e for e in edges
            if e["source"] in valid_node_ids and e["target"] in valid_node_ids
        ]
        
        # Limit JSX depth to avoid deep DOM trees
        if self.config.max_jsx_depth > 0:
            filtered_nodes, filtered_edges = self._limit_jsx_depth(
                filtered_nodes, filtered_edges, self.config.max_jsx_depth
            )
        
        return filtered_nodes, filtered_edges
    
    def _limit_jsx_depth(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_depth: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Limit JSX element tree depth to avoid overwhelming detail."""
        # Find root components
        component_ids = {n["id"] for n in nodes if n["label"] == "Component"}
        
        # BFS to find JSX elements within depth limit
        jsx_elements_to_keep = set()
        queue = [(comp_id, 0) for comp_id in component_ids]
        visited = set()
        
        while queue:
            node_id, depth = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # Find JSX children
            for edge in edges:
                if edge["source"] == node_id and edge["type"] in ("componentRendersJsx", "jsxContainsJsx", "jsxHasChild"):
                    child_id = edge["target"]
                    jsx_elements_to_keep.add(child_id)
                    if depth < max_depth:
                        queue.append((child_id, depth + 1))
        
        # Filter nodes: keep non-JSX or JSX within depth
        filtered_nodes = [
            n for n in nodes
            if n["label"] != "JSXElement" or n["id"] in jsx_elements_to_keep
        ]
        
        valid_ids = {n["id"] for n in filtered_nodes}
        filtered_edges = [
            e for e in edges
            if e["source"] in valid_ids and e["target"] in valid_ids
        ]
        
        return filtered_nodes, filtered_edges
    
    def _aggregate_styling_nodes(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Aggregate styling information instead of keeping individual nodes."""
        styling_summary = {
            "tailwind_classes": set(),
            "inline_styles": [],
            "component_styling": {}
        }
        
        # Collect styling information
        for edge in edges:
            if edge["type"] == "jsxUsesTailwindUtility":
                target_node = next((n for n in nodes if n["id"] == edge["target"]), None)
                if target_node and "name" in target_node:
                    styling_summary["tailwind_classes"].add(target_node["name"])
            elif edge["type"] == "jsxHasInlineStyle":
                target_node = next((n for n in nodes if n["id"] == edge["target"]), None)
                if target_node:
                    styling_summary["inline_styles"].append(target_node.get("properties", {}))
        
        # Remove styling nodes and edges
        styling_types = {"TailwindUtility", "InlineStyle"}
        filtered_nodes = [n for n in nodes if n["label"] not in styling_types]
        
        valid_ids = {n["id"] for n in filtered_nodes}
        filtered_edges = [
            e for e in edges
            if e["source"] in valid_ids and e["target"] in valid_ids
        ]
        
        # Convert sets to lists for JSON serialization
        styling_summary["tailwind_classes"] = sorted(list(styling_summary["tailwind_classes"]))
        
        return filtered_nodes, filtered_edges, styling_summary
    
    # --------------- Formatting & Deduplication ---------------
    def _format_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Format node into standard structure."""
        properties = node.get("props") or node.get("properties", {})
        return {
            "id": node["id"],
            "label": node.get("type", "KGNode"),
            "name": node.get("name") or node.get("path", ""),
            "properties": properties
        }
    
    def _deduplicate_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate nodes by ID."""
        seen = set()
        unique = []
        for node in nodes:
            if node["id"] not in seen:
                seen.add(node["id"])
                unique.append(node)
        return unique
    
    def _deduplicate_edges(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate edges by (source, target, type)."""
        seen = set()
        unique = []
        for edge in edges:
            key = (edge["source"], edge["target"], edge["type"])
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        return unique
    
    def _build_summary(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        styling_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build summary statistics for the subgraph."""
        node_types = {}
        edge_types = {}
        
        for node in nodes:
            node_type = node.get("label", "Unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for edge in edges:
            edge_type = edge.get("type", "Unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        summary = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_types": node_types,
            "edge_types": edge_types
        }
        
        # Add styling summary if available
        if styling_summary:
            summary["styling"] = {
                "tailwind_classes_count": len(styling_summary["tailwind_classes"]),
                "tailwind_classes": styling_summary["tailwind_classes"][:50],  # Limit for brevity
                "inline_styles_count": len(styling_summary["inline_styles"])
            }
        
        return summary


# --------------- Public API ---------------
def retrieve_subgraph(
    structured_input: Dict[str, Any],
    depth: int = 2,
    config: Optional[Stage2Config] = None
) -> Dict[str, Any]:
    """Retrieve subgraph from Neo4j for the given routes and components.
    
    Args:
        structured_input: Dict with 'routes' and 'components' lists
        depth: Max traversal depth (default: 2)
        config: Optional Stage2Config instance
    
    Returns:
        {
            "nodes": [{"id": str, "label": str, "properties": dict}],
            "edges": [{"source": str, "target": str, "type": str, "properties": dict}],
            "summary": {"node_count": int, "edge_count": int, "node_types": dict}
        }
    """
    with SubgraphRetriever(config) as retriever:
        return retriever.retrieve_subgraph(structured_input, depth)


# --------------- CLI ---------------
def _main_cli():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Stage 2A — Subgraph Retrieval (Smart Filtering)")
    parser.add_argument("--routes", nargs="+", help="Route paths (e.g., /dashboard /login)")
    parser.add_argument("--components", nargs="+", help="Component names (e.g., Dashboard Login)")
    parser.add_argument("--depth", type=int, default=2, help="Max traversal depth (default: 2)")
    parser.add_argument("--input-json", type=str, help="Path to JSON file with structured input from Stage 1")
    parser.add_argument("--output", type=str, help="Output file path (default: stdout)")
    
    # Smart filtering options
    parser.add_argument("--no-aggregate-styling", action="store_true", help="Don't aggregate styling nodes (keep individual nodes)")
    parser.add_argument("--jsx-depth", type=int, default=1, help="Max JSX tree depth (default: 1, 0=unlimited)")
    parser.add_argument("--include-file-context", action="store_true", help="Include file nodes")
    parser.add_argument("--include-only", nargs="+", help="Only include these node types (whitelist)")
    parser.add_argument("--exclude", nargs="+", help="Exclude these node types (blacklist)")
    
    args = parser.parse_args()
    
    # Build custom config based on CLI args
    cfg = Stage2Config()
    if args.no_aggregate_styling:
        cfg.aggregate_styling = False
    if args.jsx_depth is not None:
        cfg.max_jsx_depth = args.jsx_depth
    if args.include_file_context:
        cfg.include_file_context = True
    if args.include_only:
        cfg.include_only_node_types = args.include_only
        cfg.exclude_node_types = []  # Clear defaults
    elif args.exclude:
        cfg.exclude_node_types = args.exclude
    
    # Get input
    if args.input_json:
        with open(args.input_json, "r") as f:
            data = json.load(f)
            # Extract from Stage 1 output if present
            if "structured" in data:
                structured_input = data["structured"]
            else:
                structured_input = data
    elif args.routes or args.components:
        structured_input = {
            "routes": args.routes or [],
            "components": args.components or []
        }
    else:
        # Try to read from stdin
        if not sys.stdin.isatty():
            try:
                data = json.load(sys.stdin)
                if "structured" in data:
                    structured_input = data["structured"]
                else:
                    structured_input = data
            except Exception:
                parser.print_usage(sys.stderr)
                print("error: provide --routes, --components, --input-json, or pipe JSON via stdin", file=sys.stderr)
                sys.exit(2)
        else:
            parser.print_usage(sys.stderr)
            print("error: provide --routes, --components, --input-json, or pipe JSON via stdin", file=sys.stderr)
            sys.exit(2)
    
    # Retrieve subgraph with custom config
    subgraph = retrieve_subgraph(structured_input, depth=args.depth, config=cfg)
    
    # Output
    output_str = json.dumps(subgraph, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Subgraph written to {args.output}", file=sys.stderr)
    else:
        print(output_str)


if __name__ == "__main__":
    _main_cli()

