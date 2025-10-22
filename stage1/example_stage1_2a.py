"""
Example: Complete Stage 1 + Stage 2A Pipeline

Demonstrates:
1. Stage 1A: Entity extraction via similarity search
2. Stage 1B: LLM-based structured interpretation
3. Stage 2A: Subgraph retrieval from Neo4j KG

Usage:
    python example_stage1_2a.py "Register → Dashboard → Analytics"
"""

import json
import sys
from service import Stage1Service
from subgraph_retrieval import retrieve_subgraph


def run_complete_pipeline(scenario: str, depth: int = 2):
    """Run the complete Stage 1 + Stage 2A pipeline."""
    
    print(f"🔍 Processing scenario: {scenario}")
    print("=" * 80)
    
    # Stage 1: Entity Extraction + Interpretation
    print("\n📊 Stage 1A+1B: Scenario Interpretation & Entity Extraction...")
    stage1_service = Stage1Service()
    stage1_output = stage1_service.run_stage1(scenario)
    
    # Display Stage 1 results
    print(f"  ✓ Found {len(stage1_output['routes'])} routes")
    for r in stage1_output['routes'][:3]:
        print(f"    - {r['name']} (score: {r['score']:.3f})")
    
    print(f"  ✓ Found {len(stage1_output['components'])} components")
    for c in stage1_output['components'][:3]:
        print(f"    - {c['name']} (score: {c['score']:.3f})")
    
    if 'structured' in stage1_output:
        structured = stage1_output['structured']
        print(f"\n  ✓ Structured interpretation:")
        print(f"    - Type: {structured['type']}")
        print(f"    - Auth required: {structured['auth_required']}")
        print(f"    - Priority: {structured['priority']}")
        print(f"    - Goal: {structured['goal']}")
    
    # Stage 2A: Subgraph Retrieval (with smart filtering)
    print(f"\n🕸️  Stage 2A: Subgraph Retrieval (depth={depth}, smart filtering)...")
    
    # Use structured output if available, otherwise use routes/components directly
    if 'structured' in stage1_output:
        input_data = stage1_output['structured']
    else:
        input_data = {
            'routes': [r['name'] for r in stage1_output['routes']],
            'components': [c['name'] for c in stage1_output['components']]
        }
    
    # Use minimal filtering for test generation (focus on test-relevant entities)
    from subgraph_retrieval import Stage2Config
    config = Stage2Config()
    config.include_only_node_types = ["Component", "Route", "State", "EventHandler", "Hook", "Prop"]
    
    subgraph = retrieve_subgraph(input_data, depth=depth, config=config)
    
    # Display Stage 2A results
    summary = subgraph['summary']
    print(f"  ✓ Retrieved subgraph:")
    print(f"    - Total nodes: {summary['node_count']}")
    print(f"    - Total edges: {summary['edge_count']}")
    print(f"\n  📦 Node breakdown:")
    for node_type, count in sorted(summary['node_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"    - {node_type}: {count}")
    
    print(f"\n  🔗 Edge breakdown:")
    for edge_type, count in sorted(summary['edge_types'].items(), key=lambda x: -x[1])[:10]:
        print(f"    - {edge_type}: {count}")
    
    # Return combined output
    return {
        'stage1': stage1_output,
        'stage2a': subgraph
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_stage1_2a.py \"<scenario>\" [depth]")
        print("\nExamples:")
        print('  python example_stage1_2a.py "Register → Dashboard → Analytics"')
        print('  python example_stage1_2a.py "Login → Dashboard → Todos CRUD" 3')
        sys.exit(1)
    
    scenario = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    result = run_complete_pipeline(scenario, depth)
    
    # Optionally save to file
    output_file = "/tmp/pipeline_output.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Pipeline complete! Full output saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

