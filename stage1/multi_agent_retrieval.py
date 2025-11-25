"""
Multi-Agent Knowledge Graph Retrieval System

Implements intelligent, iterative retrieval from the knowledge graph using
multiple specialized agents that collaborate to gather comprehensive context.

Agents:
- SeedRetriever: Initial retrieval based on routes/components
- ContextExpander: Expands context through multi-hop traversal
- RelevanceFilter: Filters and ranks nodes by relevance
- RelationshipMapper: Maps relationships between entities

This replaces the simple single-pass retrieval in subgraph_retrieval.py
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from neo4j import GraphDatabase

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

# LLM via LiteLLM
try:
    from langchain_community.chat_models import ChatLiteLLM
    from langchain_core.messages import HumanMessage, SystemMessage
except Exception:
    ChatLiteLLM = None
    HumanMessage = None
    SystemMessage = None


class RetrievalStrategy(str, Enum):
    """Strategy for multi-hop retrieval"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    PRIORITY_GUIDED = "priority_guided"  # LLM-guided prioritization


@dataclass
class MultiAgentRetrievalConfig:
    """Configuration for multi-agent retrieval system"""
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pass: str = os.getenv("NEO4J_PASS", os.getenv("NEO4J_PASSWORD", "admin123"))
    
    # Multi-hop retrieval settings
    max_iterations: int = int(os.getenv("MULTI_AGENT_MAX_ITERATIONS", "3"))
    initial_depth: int = int(os.getenv("MULTI_AGENT_INITIAL_DEPTH", "1"))
    expansion_depth: int = int(os.getenv("MULTI_AGENT_EXPANSION_DEPTH", "2"))
    strategy: RetrievalStrategy = RetrievalStrategy(os.getenv("MULTI_AGENT_STRATEGY", "priority_guided"))
    
    # Filtering and ranking
    min_relevance_score: float = float(os.getenv("MULTI_AGENT_MIN_RELEVANCE", "0.3"))
    max_nodes_per_iteration: int = int(os.getenv("MULTI_AGENT_MAX_NODES", "100"))
    
    # LLM for intelligent guidance
    llm_model: str = os.getenv("MULTI_AGENT_LLM_MODEL", "groq/llama-3.3-70b-versatile")
    temperature: float = float(os.getenv("MULTI_AGENT_TEMPERATURE", "0.2"))
    
    # Logging
    verbose: bool = True
    
    # Node type priorities (for priority-guided strategy)
    node_type_priorities: Dict[str, float] = field(default_factory=lambda: {
        "Route": 1.0,
        "Component": 1.0,
        "State": 0.8,
        "EventHandler": 0.8,
        "Hook": 0.7,
        "Prop": 0.6,
        "Context": 0.7,
        "JSXElement": 0.3,
        "File": 0.2,
    })


@dataclass
class RetrievalContext:
    """Context maintained across retrieval iterations"""
    scenario: str
    goal: str
    
    # Accumulated nodes and edges
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tracking
    visited_node_ids: Set[str] = field(default_factory=set)
    frontier_node_ids: Set[str] = field(default_factory=set)
    iteration: int = 0
    
    # Agent decisions
    expansion_decisions: List[Dict[str, Any]] = field(default_factory=list)


class SeedRetrieverAgent:
    """Agent responsible for initial seed retrieval"""
    
    def __init__(self, config: MultiAgentRetrievalConfig, driver):
        self.config = config
        self.driver = driver
    
    def retrieve_seeds(
        self, 
        routes: List[str], 
        components: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve initial seed nodes from routes and components"""
        
        print(f"[SeedRetriever] Retrieving seeds: {len(routes)} routes, {len(components)} components")
        
        nodes = []
        edges = []
        
        with self.driver.session() as session:
            # Retrieve route nodes
            if routes:
                route_result = session.run("""
                    MATCH (r:KGNode)
                    WHERE r.type = 'Route' AND (r.path IN $routes OR r.name IN $routes)
                    RETURN r.id AS id, r.type AS type, r.name AS name, r.path AS path, properties(r) AS props
                """, {"routes": routes})
                
                for record in route_result:
                    nodes.append(self._format_node(record))
            
            # Retrieve component nodes
            if components:
                comp_result = session.run("""
                    MATCH (c:KGNode)
                    WHERE c.type = 'Component' AND c.name IN $components
                    RETURN c.id AS id, c.type AS type, c.name AS name, properties(c) AS props
                """, {"components": components})
                
                for record in comp_result:
                    nodes.append(self._format_node(record))
        
        print(f"[SeedRetriever] Retrieved {len(nodes)} seed nodes")
        
        return nodes, edges
    
    def _format_node(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Format node into standard structure"""
        return {
            "id": record.get("id", ""),
            "label": record.get("type") or "KGNode",
            "name": record.get("name") or record.get("path") or "",
            "properties": record.get("props") or {}
        }


class ContextExpanderAgent:
    """Agent responsible for expanding context through multi-hop traversal"""
    
    def __init__(self, config: MultiAgentRetrievalConfig, driver):
        self.config = config
        self.driver = driver
    
    def expand_context(
        self, 
        ctx: RetrievalContext,
        expansion_strategy: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Expand context from frontier nodes"""
        
        if not ctx.frontier_node_ids:
            print(f"[ContextExpander] No frontier nodes to expand")
            return [], []
        
        frontier_list = list(ctx.frontier_node_ids)[:50]  # Limit frontier size
        print(f"[ContextExpander] Expanding from {len(frontier_list)} frontier nodes")
        
        new_nodes = []
        new_edges = []
        
        with self.driver.session() as session:
            # Multi-hop expansion query
            result = session.run(f"""
                MATCH (seed:KGNode)
                WHERE seed.id IN $seed_ids
                OPTIONAL MATCH path = (seed)-[r:KGR*1..{self.config.expansion_depth}]-(connected:KGNode)
                WHERE NOT connected.id IN $visited_ids
                WITH collect(DISTINCT connected) AS newNodes, 
                     collect(DISTINCT r) AS rels
                UNWIND newNodes AS n
                WITH n WHERE n IS NOT NULL
                RETURN DISTINCT n.id AS id, n.type AS type, n.name AS name, n.path AS path, properties(n) AS props
                LIMIT {self.config.max_nodes_per_iteration}
            """, {
                "seed_ids": frontier_list,
                "visited_ids": list(ctx.visited_node_ids)
            })
            
            for record in result:
                node = self._format_node(record)
                if node["id"] not in ctx.visited_node_ids:
                    new_nodes.append(node)
            
            # Get edges between all nodes (old + new)
            if new_nodes:
                all_node_ids = list(ctx.visited_node_ids) + [n["id"] for n in new_nodes]
                edge_result = session.run("""
                    MATCH (from:KGNode)-[r:KGR]->(to:KGNode)
                    WHERE from.id IN $node_ids AND to.id IN $node_ids
                    RETURN DISTINCT from.id AS source, to.id AS target, r.type AS type, properties(r) AS props
                """, {"node_ids": all_node_ids})
                
                for record in edge_result:
                    edge = {
                        "source": record["source"],
                        "target": record["target"],
                        "type": record["type"],
                        "properties": record.get("props", {})
                    }
                    new_edges.append(edge)
        
        print(f"[ContextExpander] Expanded to {len(new_nodes)} new nodes, {len(new_edges)} total edges")
        
        return new_nodes, new_edges
    
    def _format_node(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Format node into standard structure"""
        return {
            "id": record.get("id", ""),
            "label": record.get("type") or "KGNode",
            "name": record.get("name") or record.get("path") or "",
            "properties": record.get("props") or {}
        }


class RelevanceFilterAgent:
    """Agent responsible for filtering and ranking nodes by relevance"""
    
    def __init__(self, config: MultiAgentRetrievalConfig):
        self.config = config
    
    def filter_and_rank(
        self,
        nodes: List[Dict[str, Any]],
        scenario: str,
        goal: str
    ) -> List[Dict[str, Any]]:
        """Filter nodes and add relevance scores"""
        
        if self.config.verbose:
            print(f"[RelevanceFilter] Filtering {len(nodes)} nodes")
        
        scored_nodes = []
        
        for node in nodes:
            score = self._compute_relevance_score(node, scenario, goal)
            
            if score >= self.config.min_relevance_score:
                node["relevance_score"] = score
                scored_nodes.append(node)
        
        # Sort by relevance
        scored_nodes.sort(key=lambda n: n.get("relevance_score", 0), reverse=True)
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"🔍 RELEVANCE FILTER AGENT - SCORING & FILTERING")
            print(f"{'='*70}")
            print(f"📊 Analysis:")
            print(f"   • Nodes analyzed: {len(nodes)}")
            print(f"   • Passed threshold ({self.config.min_relevance_score}): {len(scored_nodes)}")
            print(f"   • Filtered out: {len(nodes) - len(scored_nodes)} low-relevance nodes")
            
            if scored_nodes:
                print(f"\n🏆 Top 5 Most Relevant Nodes:")
                for i, node in enumerate(scored_nodes[:5], 1):
                    print(f"   {i}. {node.get('label', 'Unknown'):15} | {node.get('name', 'N/A')[:35]:35} | Score: {node['relevance_score']:.2f}")
                
                # Show node type distribution
                type_counts = {}
                for node in scored_nodes:
                    node_type = node.get('label', 'Unknown')
                    type_counts[node_type] = type_counts.get(node_type, 0) + 1
                
                print(f"\n📈 Filtered Node Types:")
                sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
                for node_type, count in sorted_types[:5]:
                    print(f"   • {node_type}: {count}")
            
            print(f"{'='*70}\n")
        else:
            print(f"[RelevanceFilter] Kept {len(scored_nodes)} nodes above threshold")
        
        return scored_nodes
    
    def _compute_relevance_score(
        self,
        node: Dict[str, Any],
        scenario: str,
        goal: str
    ) -> float:
        """Compute relevance score for a node"""
        
        # Base score from node type priority (handle None values)
        node_type = node.get("label") or ""
        base_score = self.config.node_type_priorities.get(node_type, 0.5)
        
        # Keyword matching bonus (handle None values)
        name = (node.get("name") or "").lower()
        scenario_lower = (scenario or "").lower()
        goal_lower = (goal or "").lower()
        
        keyword_bonus = 0.0
        
        # Check if node name appears in scenario/goal
        if name and len(name) > 2:
            if name in scenario_lower:
                keyword_bonus += 0.3
            if name in goal_lower:
                keyword_bonus += 0.2
        
        # Check individual words
        words = name.split()
        for word in words:
            if len(word) > 3:
                if word in scenario_lower:
                    keyword_bonus += 0.1
                if word in goal_lower:
                    keyword_bonus += 0.05
        
        return min(1.0, base_score + keyword_bonus)


class PriorityGuidanceAgent:
    """Agent that uses LLM to guide expansion priorities"""
    
    def __init__(self, config: MultiAgentRetrievalConfig):
        self.config = config
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError("LangChain dependencies required for PriorityGuidanceAgent")
    
    def prioritize_expansion(
        self,
        ctx: RetrievalContext,
        candidate_nodes: List[Dict[str, Any]]
    ) -> List[str]:
        """Use LLM to prioritize which nodes to expand next"""
        
        if not candidate_nodes:
            return []
        
        print(f"[PriorityGuidance] Prioritizing {len(candidate_nodes)} candidate nodes")
        
        try:
            llm = ChatLiteLLM(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=500
            )
            
            # Build summary of candidates
            candidates_summary = []
            for node in candidate_nodes[:20]:  # Limit to prevent token overflow
                candidates_summary.append({
                    "id": node.get("id", ""),
                    "type": node.get("label") or "",
                    "name": node.get("name") or "",
                    "relevance": node.get("relevance_score", 0.5)
                })
            
            system_prompt = """You are an intelligent knowledge graph navigator for test generation.
Your job is to select the most relevant nodes to expand next based on the test scenario.

Consider:
1. Relevance to the user scenario and goal
2. Node types that provide actionable test context (States, EventHandlers, Hooks)
3. Coverage of the user flow
4. Avoiding low-value nodes (styling, purely structural elements)

Return a JSON object with:
- "reasoning": Brief explanation of why you selected these nodes
- "node_ids": Array of node IDs to expand, ordered by priority (most important first)
Limit: 10 nodes maximum.

Output format: {"reasoning": "explanation...", "node_ids": ["node_id_1", "node_id_2", ...]}
"""
            
            user_prompt = f"""Scenario: {ctx.scenario}
Goal: {ctx.goal}
Iteration: {ctx.iteration}

Candidate nodes to consider:
{json.dumps(candidates_summary, indent=2)}

Which nodes should we expand next? Return JSON array of node IDs:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            
            # Try to extract JSON object with reasoning
            try:
                # Look for JSON object
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    result = json.loads(content[start:end+1])
                    reasoning = result.get("reasoning", "No reasoning provided")
                    node_ids = result.get("node_ids", [])
                    
                    if self.config.verbose:
                        print(f"\n{'='*70}")
                        print(f"🧠 PRIORITY GUIDANCE AGENT - THINKING PROCESS")
                        print(f"{'='*70}")
                        print(f"📊 Context:")
                        print(f"   • Scenario: {ctx.scenario[:60]}...")
                        print(f"   • Goal: {ctx.goal[:60]}...")
                        print(f"   • Iteration: {ctx.iteration}")
                        print(f"   • Candidates analyzed: {len(candidate_nodes)}")
                        print(f"\n💭 Agent Reasoning:")
                        print(f"   {reasoning}")
                        print(f"\n✅ Selected {len(node_ids)} nodes for expansion:")
                        for i, nid in enumerate(node_ids[:5], 1):
                            matching_node = next((n for n in candidate_nodes if n.get("id") == nid), None)
                            if matching_node:
                                print(f"   {i}. {matching_node.get('label', 'Unknown'):15} | {matching_node.get('name', 'N/A')[:35]:35} | Score: {matching_node.get('relevance_score', 0):.2f}")
                        if len(node_ids) > 5:
                            print(f"   ... and {len(node_ids) - 5} more nodes")
                        print(f"{'='*70}\n")
                    else:
                        print(f"[PriorityGuidance] LLM selected {len(node_ids)} nodes for expansion")
                    
                    return node_ids[:10]
                    
                # Fallback to array format
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1:
                    node_ids = json.loads(content[start:end+1])
                    print(f"[PriorityGuidance] LLM selected {len(node_ids)} nodes (no reasoning provided)")
                    return node_ids[:10]
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            print(f"[PriorityGuidance] LLM guidance failed: {e}, using fallback")
        
        # Fallback: take top nodes by relevance
        sorted_nodes = sorted(
            candidate_nodes,
            key=lambda n: n.get("relevance_score", 0),
            reverse=True
        )
        return [n["id"] for n in sorted_nodes[:10]]


class MultiAgentRetriever:
    """Orchestrator for multi-agent knowledge graph retrieval"""
    
    def __init__(self, config: Optional[MultiAgentRetrievalConfig] = None):
        self.config = config or MultiAgentRetrievalConfig()
        
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_pass),
        )
        
        # Initialize agents
        self.seed_retriever = SeedRetrieverAgent(self.config, self.driver)
        self.context_expander = ContextExpanderAgent(self.config, self.driver)
        self.relevance_filter = RelevanceFilterAgent(self.config)
        
        # Optional: priority guidance agent
        self.priority_guidance = None
        if self.config.strategy == RetrievalStrategy.PRIORITY_GUIDED:
            try:
                self.priority_guidance = PriorityGuidanceAgent(self.config)
            except RuntimeError:
                print("[MultiAgentRetriever] Priority guidance unavailable, falling back to relevance-based")
                self.config.strategy = RetrievalStrategy.BREADTH_FIRST
    
    def close(self):
        self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def retrieve(
        self,
        structured_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute multi-agent retrieval workflow"""
        
        print(f"\n{'='*80}")
        print("🤖 Multi-Agent Knowledge Graph Retrieval")
        print(f"{'='*80}")
        print(f"Strategy: {self.config.strategy.value}")
        print(f"Max Iterations: {self.config.max_iterations}")
        print(f"{'='*80}\n")
        
        # Extract inputs
        routes = structured_input.get("routes", [])
        components = structured_input.get("components", [])
        scenario = structured_input.get("goal", "")
        goal = scenario
        
        # Initialize context
        ctx = RetrievalContext(
            scenario=scenario,
            goal=goal
        )
        
        # Step 1: Seed retrieval
        print(f"\n[Iteration 0] Seed Retrieval")
        seed_nodes, seed_edges = self.seed_retriever.retrieve_seeds(routes, components)
        
        if not seed_nodes:
            print("⚠️  No seed nodes found")
            return self._build_output(ctx)
        
        ctx.nodes.extend(seed_nodes)
        ctx.edges.extend(seed_edges)
        for node in seed_nodes:
            ctx.visited_node_ids.add(node["id"])
            ctx.frontier_node_ids.add(node["id"])
        
        # Step 2: Iterative expansion
        for iteration in range(1, self.config.max_iterations + 1):
            ctx.iteration = iteration
            
            print(f"\n[Iteration {iteration}] Context Expansion")
            print(f"  Frontier size: {len(ctx.frontier_node_ids)}")
            print(f"  Total nodes: {len(ctx.nodes)}")
            
            # Expand context
            new_nodes, new_edges = self.context_expander.expand_context(ctx)
            
            if not new_nodes:
                print(f"  No new nodes found, stopping expansion")
                break
            
            # Filter and rank new nodes
            filtered_nodes = self.relevance_filter.filter_and_rank(
                new_nodes,
                scenario,
                goal
            )
            
            # Decide which nodes to add to frontier for next iteration
            if self.priority_guidance and iteration < self.config.max_iterations:
                priority_node_ids = self.priority_guidance.prioritize_expansion(
                    ctx,
                    filtered_nodes
                )
                next_frontier = set(priority_node_ids)
            else:
                # Take top nodes by relevance
                next_frontier = {n["id"] for n in filtered_nodes[:15]}
            
            # Update context
            ctx.nodes.extend(filtered_nodes)
            ctx.edges = new_edges  # Replace with full edge list
            for node in filtered_nodes:
                ctx.visited_node_ids.add(node["id"])
            ctx.frontier_node_ids = next_frontier
            
            ctx.expansion_decisions.append({
                "iteration": iteration,
                "new_nodes": len(filtered_nodes),
                "next_frontier_size": len(next_frontier)
            })
            
            print(f"  Added {len(filtered_nodes)} nodes")
            print(f"  Next frontier: {len(next_frontier)} nodes")
        
        # Build final output
        return self._build_output(ctx)
    
    def _build_output(self, ctx: RetrievalContext) -> Dict[str, Any]:
        """Build final output structure"""
        
        # Deduplicate edges
        edge_keys = set()
        unique_edges = []
        for edge in ctx.edges:
            key = (edge["source"], edge["target"], edge["type"])
            if key not in edge_keys:
                edge_keys.add(key)
                unique_edges.append(edge)
        
        # Build summary
        node_types = {}
        for node in ctx.nodes:
            node_type = node.get("label", "Unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        edge_types = {}
        for edge in unique_edges:
            edge_type = edge.get("type", "Unknown")
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        summary = {
            "node_count": len(ctx.nodes),
            "edge_count": len(unique_edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "iterations": ctx.iteration,
            "expansion_decisions": ctx.expansion_decisions
        }
        
        print(f"\n{'='*80}")
        print("✅ Multi-Agent Retrieval Complete")
        print(f"{'='*80}")
        print(f"Total nodes: {len(ctx.nodes)}")
        print(f"Total edges: {len(unique_edges)}")
        print(f"Iterations: {ctx.iteration}")
        print(f"Node types: {', '.join([f'{k}({v})' for k, v in sorted(node_types.items(), key=lambda x: -x[1])[:5]])}")
        print(f"{'='*80}\n")
        
        return {
            "nodes": ctx.nodes,
            "edges": unique_edges,
            "summary": summary
        }


# --------------- Public API ---------------
def multi_agent_retrieve(
    structured_input: Dict[str, Any],
    config: Optional[MultiAgentRetrievalConfig] = None
) -> Dict[str, Any]:
    """Execute multi-agent knowledge graph retrieval
    
    Args:
        structured_input: Dict with 'routes', 'components', 'goal'
        config: Optional MultiAgentRetrievalConfig
    
    Returns:
        {
            "nodes": List[Dict],
            "edges": List[Dict],
            "summary": Dict
        }
    """
    with MultiAgentRetriever(config) as retriever:
        return retriever.retrieve(structured_input)


# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Multi-Agent Knowledge Graph Retrieval")
    parser.add_argument("--input-json", type=str, help="Path to JSON file with structured input")
    parser.add_argument("--routes", nargs="+", help="Route paths")
    parser.add_argument("--components", nargs="+", help="Component names")
    parser.add_argument("--goal", type=str, default="", help="Test goal/scenario description")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations (default: 3)")
    parser.add_argument("--strategy", type=str, choices=["breadth_first", "priority_guided"], 
                       default="priority_guided", help="Retrieval strategy")
    
    args = parser.parse_args()
    
    # Build config
    config = MultiAgentRetrievalConfig()
    config.max_iterations = args.iterations
    config.strategy = RetrievalStrategy(args.strategy)
    
    # Get input
    if args.input_json:
        with open(args.input_json, "r") as f:
            structured_input = json.load(f)
            if "structured" in structured_input:
                structured_input = structured_input["structured"]
    elif args.routes or args.components:
        structured_input = {
            "routes": args.routes or [],
            "components": args.components or [],
            "goal": args.goal
        }
    else:
        if not sys.stdin.isatty():
            try:
                structured_input = json.load(sys.stdin)
                if "structured" in structured_input:
                    structured_input = structured_input["structured"]
            except Exception:
                parser.print_usage(sys.stderr)
                print("error: provide --input-json, --routes/--components, or pipe JSON via stdin", file=sys.stderr)
                sys.exit(2)
        else:
            parser.print_usage(sys.stderr)
            print("error: provide --input-json, --routes/--components, or pipe JSON via stdin", file=sys.stderr)
            sys.exit(2)
    
    # Execute retrieval
    result = multi_agent_retrieve(structured_input, config)
    
    # Output
    output_str = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"✅ Output saved: {args.output}", file=sys.stderr)
    else:
        print(output_str)

