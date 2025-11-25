"""
Example: Multi-Agent Test Generation Pipeline

This script demonstrates the new multi-agent system with:
- Intelligent multi-hop knowledge graph retrieval
- Test planning and architecture design
- Iterative test generation with quality refinement

Compare this to example_end_to_end.py to see the improvements!
"""

import json
import sys


def example_basic():
    """Basic example with default settings"""
    
    from multi_agent_pipeline import run_multi_agent_pipeline
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Multi-Agent Pipeline")
    print("="*80 + "\n")
    
    scenario = "User logs in → Views dashboard → Creates a new todo item"
    
    result = run_multi_agent_pipeline(
        scenario=scenario,
        retrieval_iterations=3,
        refinement_iterations=2,
        output_dir="/tmp/multi_agent_example1"
    )
    
    # Print key results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n✅ Stage 1: Found {len(result['stage1']['routes'])} routes, {len(result['stage1']['components'])} components")
    print(f"✅ Stage 2: Retrieved {result['stage2']['summary']['node_count']} nodes in {result['stage2']['summary']['iterations']} iterations")
    print(f"✅ Stage 3: Generated tests with quality score {result['stage3']['metadata'].get('final_quality_score', 'N/A')}")
    
    if result['stage3'].get('test_specs'):
        print(f"\nGenerated {len(result['stage3']['test_specs'])} test cases:")
        for spec in result['stage3']['test_specs'][:3]:
            print(f"  - {spec['id']}: {spec['title']}")
    
    return result


def example_cypress_specs():
    """Example generating Cypress test specs only"""
    
    from multi_agent_pipeline import run_multi_agent_pipeline
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Cypress Test Specs Generation")
    print("="*80 + "\n")
    
    scenario = "User registers → Verifies email → Completes profile setup"
    
    result = run_multi_agent_pipeline(
        scenario=scenario,
        test_framework="cypress",
        output_format="spec",
        retrieval_iterations=2,
        refinement_iterations=3,
        quality_threshold=0.85,
        output_dir="/tmp/multi_agent_example2"
    )
    
    # Show test specifications
    if result['stage3'].get('test_specs'):
        print("\n" + "="*80)
        print("GENERATED TEST SPECIFICATIONS")
        print("="*80 + "\n")
        
        for spec in result['stage3']['test_specs']:
            print(f"\n{spec['id']}: {spec['title']}")
            print(f"Priority: {spec.get('priority', 'medium')}")
            print(f"Coverage: {spec.get('coverage_type', 'N/A')}")
            print(f"Steps:")
            for i, step in enumerate(spec['steps'], 1):
                print(f"  {i}. {step}")
            print(f"Expected: {spec['expected_result']}")
            if 'rationale' in spec:
                print(f"Rationale: {spec['rationale'][:100]}...")
            print()
    
    return result


def example_playwright_code():
    """Example generating Playwright test code with high quality threshold"""
    
    from multi_agent_pipeline import run_multi_agent_pipeline
    
    print("\n" + "="*80)
    print("EXAMPLE 3: High-Quality Playwright Test Code")
    print("="*80 + "\n")
    
    scenario = "Admin logs in → Manages users → Creates new user → Assigns roles"
    
    result = run_multi_agent_pipeline(
        scenario=scenario,
        test_framework="playwright",
        output_format="code",
        retrieval_iterations=4,
        retrieval_strategy="priority_guided",
        refinement_iterations=3,
        quality_threshold=0.9,  # High quality threshold
        output_dir="/tmp/multi_agent_example3"
    )
    
    # Show test code preview
    if result['stage3'].get('test_code'):
        print("\n" + "="*80)
        print("GENERATED TEST CODE (PREVIEW)")
        print("="*80 + "\n")
        
        code = result['stage3']['test_code']
        lines = code.split('\n')
        preview_lines = min(50, len(lines))
        print('\n'.join(lines[:preview_lines]))
        if len(lines) > preview_lines:
            print(f"\n... ({len(lines) - preview_lines} more lines)")
    
    # Show quality evolution
    if result['stage3'].get('reviews'):
        print("\n" + "="*80)
        print("QUALITY EVOLUTION")
        print("="*80 + "\n")
        
        for review in result['stage3']['reviews']:
            it = review.get('iteration', 0)
            score = review.get('score', 0)
            print(f"Iteration {it}: Quality Score = {score:.2f}")
            if review.get('weaknesses'):
                print(f"  Weaknesses: {', '.join(review['weaknesses'][:2])}")
            if review.get('suggestions'):
                print(f"  Suggestions: {', '.join(review['suggestions'][:2])}")
            print()
    
    return result


def example_comparison():
    """Compare old vs new pipeline"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Old vs New Pipeline Comparison")
    print("="*80 + "\n")
    
    scenario = "User searches for products → Filters by category → Adds to cart"
    
    print("Running OLD pipeline (simple single-pass)...")
    print("-" * 40)
    
    # Old pipeline (existing example_end_to_end.py approach)
    from service import Stage1Service
    from subgraph_retrieval import retrieve_subgraph, Stage2Config
    from test_generator import generate_test, Stage3Config, TestFramework
    
    # Stage 1
    stage1_service = Stage1Service()
    stage1_old = stage1_service.run_stage1(scenario)
    
    # Stage 2 (simple retrieval)
    config2_old = Stage2Config()
    config2_old.include_only_node_types = ["Component", "Route", "State", "EventHandler"]
    subgraph_old = retrieve_subgraph(stage1_old['structured'], depth=2, config=config2_old)
    
    # Stage 3 (single-pass generation)
    config3_old = Stage3Config()
    config3_old.test_framework = TestFramework("playwright")
    test_old = generate_test(stage1_old['structured'], subgraph_old, config3_old)
    
    print(f"✓ Old pipeline: {subgraph_old['summary']['node_count']} nodes, single test generation pass")
    print(f"  Assertions: {test_old['metadata']['assertions_count']}")
    print(f"  Code length: {len(test_old['test_code'])} chars")
    
    print("\nRunning NEW multi-agent pipeline...")
    print("-" * 40)
    
    # New pipeline
    from multi_agent_pipeline import run_multi_agent_pipeline
    
    result_new = run_multi_agent_pipeline(
        scenario=scenario,
        retrieval_iterations=3,
        refinement_iterations=2,
        output_dir="/tmp/multi_agent_comparison",
        save_intermediate=False,
        verbose=False
    )
    
    print(f"✓ New pipeline: {result_new['stage2']['summary']['node_count']} nodes with {result_new['stage2']['summary']['iterations']} retrieval iterations")
    print(f"  Test scenarios planned: {len(result_new['stage3'].get('test_plan', {}).get('scenarios', []))}")
    print(f"  Test specs generated: {len(result_new['stage3'].get('test_specs', []))}")
    print(f"  Refinement iterations: {result_new['stage3']['metadata']['refinement_iterations']}")
    print(f"  Final quality score: {result_new['stage3']['metadata'].get('final_quality_score', 'N/A')}")
    if result_new['stage3'].get('test_code'):
        print(f"  Code length: {len(result_new['stage3']['test_code'])} chars")
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print("\nOLD PIPELINE:")
    print("  ✗ Single-pass KG retrieval (no iteration)")
    print("  ✗ No test planning or architecture design")
    print("  ✗ Single LLM call for test generation")
    print("  ✗ No quality review or refinement")
    print("  ✗ Basic test coverage")
    
    print("\nNEW MULTI-AGENT PIPELINE:")
    print("  ✓ Multi-hop intelligent KG retrieval with expansion")
    print("  ✓ Planner Agent creates comprehensive test strategy")
    print("  ✓ Test Architect designs test structure")
    print("  ✓ Test Writer generates high-quality tests")
    print("  ✓ Reviewer Agent evaluates quality")
    print("  ✓ Iterative refinement until quality threshold met")
    print("  ✓ Comprehensive test coverage (happy path, errors, edge cases)")
    
    return result_new


def example_all():
    """Run all examples"""
    
    print("\n" + "="*80)
    print("MULTI-AGENT TEST GENERATION - ALL EXAMPLES")
    print("="*80)
    
    try:
        print("\n[1/4] Running basic example...")
        example_basic()
        
        print("\n[2/4] Running Cypress specs example...")
        example_cypress_specs()
        
        print("\n[3/4] Running Playwright code example...")
        example_playwright_code()
        
        print("\n[4/4] Running comparison example...")
        example_comparison()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Pipeline Examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["basic", "cypress", "playwright", "comparison", "all"],
        default="basic",
        help="Which example to run (default: basic)"
    )
    
    args = parser.parse_args()
    
    examples = {
        "basic": example_basic,
        "cypress": example_cypress_specs,
        "playwright": example_playwright_code,
        "comparison": example_comparison,
        "all": example_all
    }
    
    try:
        result = examples[args.example]()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


