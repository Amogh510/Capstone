"""
Complete End-to-End Pipeline: Stage 1 + 2A + 3

Demonstrates the full test case generation pipeline:
1. Stage 1A+1B: Scenario interpretation and entity extraction
2. Stage 2A: Subgraph retrieval with smart filtering
3. Stage 3: E2E test case generation

Usage:
    python example_end_to_end.py "User registers → Logs in → Views analytics dashboard"
    
    # With custom options
    python example_end_to_end.py "Login → Dashboard → Create Todo" \
        --framework cypress \
        --depth 2 \
        --output ./tests/todo-workflow.spec.js
"""

import json
import sys
import os
from typing import Optional

def run_complete_pipeline(
    scenario: str,
    depth: int = 2,
    framework: str = "playwright",
    output_format: str = "code",
    num_specs: int = 8,
    output_file: Optional[str] = None
) -> dict:
    """Run the complete Stage 1 + 2A + 3 pipeline.
    
    Args:
        scenario: User flow description
        depth: Graph traversal depth for Stage 2A
        framework: Test framework (playwright/cypress/selenium)
        output_file: Optional file path to save generated test
    
    Returns:
        Complete pipeline output including all stages
    """
    
    print(f"🚀 Starting End-to-End Test Generation Pipeline")
    print(f"📝 Scenario: {scenario}")
    print("=" * 80)
    
    # ==================== Stage 1: Scenario Interpretation ====================
    print("\n📊 Stage 1A+1B: Scenario Interpretation & Entity Extraction...")
    
    from service import Stage1Service
    stage1_service = Stage1Service()
    stage1_output = stage1_service.run_stage1(scenario)
    
    # Display Stage 1 results
    print(f"  ✓ Found {len(stage1_output['routes'])} routes:")
    for r in stage1_output['routes'][:5]:
        print(f"    - {r['name']} (score: {r['score']:.3f})")
    
    print(f"  ✓ Found {len(stage1_output['components'])} components:")
    for c in stage1_output['components'][:5]:
        print(f"    - {c['name']} (score: {c['score']:.3f})")
    
    if 'structured' in stage1_output:
        structured = stage1_output['structured']
        print(f"\n  ✓ Structured interpretation:")
        print(f"    - Type: {structured['type']}")
        print(f"    - Auth required: {structured['auth_required']}")
        print(f"    - Priority: {structured['priority']}")
        print(f"    - Goal: {structured['goal']}")
    else:
        print("  ⚠️  No structured interpretation available (Stage 1B disabled or failed)")
        # Create a minimal structured output for Stage 2A
        structured = {
            "type": "workflow",
            "routes": [r['name'] for r in stage1_output['routes'][:3]],
            "components": [c['name'] for c in stage1_output['components'][:3]],
            "auth_required": False,
            "goal": scenario,
            "priority": "medium"
        }
        stage1_output['structured'] = structured
    
    # ==================== Stage 2A: Subgraph Retrieval ====================
    print(f"\n🕸️  Stage 2A: Subgraph Retrieval (depth={depth}, minimal filtering)...")
    
    from subgraph_retrieval import retrieve_subgraph, Stage2Config
    
    # Use minimal filtering for test generation (focus on test-relevant entities)
    config = Stage2Config()
    config.include_only_node_types = ["Component", "Route", "State", "EventHandler", "Hook", "Prop"]
    
    input_data = stage1_output['structured']
    subgraph = retrieve_subgraph(input_data, depth=depth, config=config)
    
    # Display Stage 2A results
    summary = subgraph.get('summary', {})
    node_count = summary.get('node_count', 0)
    edge_count = summary.get('edge_count', 0)
    
    print(f"  ✓ Retrieved subgraph:")
    print(f"    - Total nodes: {node_count}")
    print(f"    - Total edges: {edge_count}")
    
    if node_count > 0:
        print(f"\n  📦 Node breakdown:")
        node_types = summary.get('node_types', {})
        for node_type, count in sorted(node_types.items(), key=lambda x: -x[1])[:8]:
            print(f"    - {node_type}: {count}")
    else:
        print("  ⚠️  Warning: No nodes retrieved. Check if Neo4j contains data for the scenario.")
        print("  💡 Tip: The components/routes might not exist in the knowledge graph.")
    
    # ==================== Stage 3: Test Generation ====================
    test_result = None
    test_specs = None
    
    if output_format in ["code", "both"]:
        print(f"\n🧪 Stage 3: Generating {framework.upper()} test code...")
        
        from test_generator import generate_test, Stage3Config, TestFramework
        
        # Configure test generation
        test_config = Stage3Config()
        test_config.test_framework = TestFramework(framework)
        
        try:
            test_result = generate_test(structured, subgraph, test_config)
            
            print(f"  ✓ Test code generated successfully:")
            print(f"    - Test name: {test_result['test_name']}")
            print(f"    - Framework: {test_result['framework']}")
            print(f"    - Assertions: {test_result['metadata']['assertions_count']}")
            print(f"    - Steps: {test_result['metadata']['steps_count']}")
            
        except Exception as e:
            print(f"  ❌ Test code generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    if output_format in ["spec", "both"]:
        print(f"\n📋 Stage 3: Generating test case specifications (JSON)...")
        
        from test_spec_generator import generate_test_specs, TestSpecConfig
        
        # Configure spec generation
        spec_config = TestSpecConfig()
        spec_config.num_test_cases = num_specs
        
        try:
            test_specs = generate_test_specs(structured, subgraph, spec_config)
            
            print(f"  ✓ Test specifications generated successfully:")
            print(f"    - Number of test cases: {len(test_specs)}")
            print(f"    - Coverage: Happy path, errors, edge cases, accessibility")
            
        except Exception as e:
            print(f"  ❌ Test specification generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== Output ====================
    
    # Build complete pipeline output
    pipeline_output = {
        "scenario": scenario,
        "stage1": stage1_output,
        "stage2a": subgraph,
        "stage3": {}
    }
    
    if test_result:
        pipeline_output["stage3"]["test_code"] = test_result
    if test_specs:
        pipeline_output["stage3"]["test_specs"] = test_specs
    
    # Save to file
    output_json = output_file.rsplit(".", 1)[0] + "_pipeline.json" if output_file else "/tmp/e2e_pipeline_output.json"
    with open(output_json, "w") as f:
        json.dump(pipeline_output, f, indent=2)
    
    print(f"\n📄 Pipeline output saved: {output_json}")
    
    # Save test code to separate file
    if test_result and output_file:
        with open(output_file, "w") as f:
            f.write(test_result["test_code"])
        print(f"✅ Test code saved: {output_file}")
        
        # Display preview
        print(f"\n{'='*80}")
        print("📝 Generated Test Code Preview:")
        print(f"{'='*80}")
        lines = test_result["test_code"].split("\n")
        preview_lines = min(30, len(lines))
        print("\n".join(lines[:preview_lines]))
        if len(lines) > preview_lines:
            print(f"\n... ({len(lines) - preview_lines} more lines)")
    elif test_result:
        # No output file specified, show full test
        print(f"\n{'='*80}")
        print("📝 Generated Test Code:")
        print(f"{'='*80}\n")
        print(test_result["test_code"])
    
    # Save test specs to separate JSON file
    if test_specs:
        if output_file:
            spec_file = output_file.rsplit(".", 1)[0] + "_specs.json"
        else:
            spec_file = "/tmp/test_specifications.json"
        
        with open(spec_file, "w") as f:
            json.dump(test_specs, f, indent=2)
        print(f"✅ Test specifications saved: {spec_file}")
        
        # Display preview
        print(f"\n{'='*80}")
        print("📋 Generated Test Specifications Preview:")
        print(f"{'='*80}")
        for spec in test_specs[:3]:
            print(f"\n{spec['id']}: {spec['title']}")
            print(f"  Steps: {len(spec['steps'])}")
            print(f"  Expected: {spec['expected_result'][:80]}...")
        if len(test_specs) > 3:
            print(f"\n... ({len(test_specs) - 3} more test cases)")
    
    print(f"\n{'='*80}")
    print("✅ Pipeline Complete!")
    print(f"{'='*80}\n")
    
    return pipeline_output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete E2E Test Generation Pipeline (Stage 1 + 2A + 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python example_end_to_end.py "Login → Dashboard → Analytics"
  
  # With Cypress framework
  python example_end_to_end.py "Register → Create Todo → Mark Complete" --framework cypress
  
  # Save to file
  python example_end_to_end.py "Login flow" --output ./tests/login.spec.ts
  
  # Custom depth
  python example_end_to_end.py "Complex workflow" --depth 3 --framework playwright
        """
    )
    
    parser.add_argument(
        "scenario",
        type=str,
        help="User flow scenario (e.g., 'Login → Dashboard → Analytics')"
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Graph traversal depth for Stage 2A (default: 2)"
    )
    
    parser.add_argument(
        "--framework",
        type=str,
        choices=["playwright", "cypress", "selenium"],
        default="playwright",
        help="Test framework to generate (default: playwright)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["code", "spec", "both"],
        default="code",
        help="Output format: 'code' for executable tests, 'spec' for JSON test cases, 'both' for both (default: code)"
    )
    
    parser.add_argument(
        "--num-specs",
        type=int,
        default=8,
        help="Number of test specifications to generate (default: 8)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for generated test (e.g., ./tests/scenario.spec.ts)"
    )
    
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save complete pipeline output as JSON (default: /tmp/e2e_pipeline_output.json)"
    )
    
    args = parser.parse_args()
    
    # Validate output file extension if provided
    if args.output:
        ext = args.output.rsplit(".", 1)[-1] if "." in args.output else ""
        if args.framework == "playwright" and ext not in ["ts", "js"]:
            print(f"Warning: Playwright tests typically use .spec.ts or .spec.js extension")
        elif args.framework == "cypress" and ext not in ["ts", "js", "cy.ts", "cy.js"]:
            print(f"Warning: Cypress tests typically use .spec.js or .cy.js extension")
        elif args.framework == "selenium" and ext != "py":
            print(f"Warning: Selenium tests typically use .py extension")
    
    try:
        result = run_complete_pipeline(
            scenario=args.scenario,
            depth=args.depth,
            framework=args.framework,
            output_format=args.format,
            num_specs=args.num_specs,
            output_file=args.output
        )
        
        # Save JSON if requested
        if args.save_json:
            with open(args.save_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"📦 Complete pipeline JSON saved: {args.save_json}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

