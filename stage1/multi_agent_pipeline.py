"""
Complete Multi-Agent Pipeline: Orchestrating the Full Workflow

This is the main entry point for the enhanced multi-agent test generation system.
It orchestrates all agents across the entire pipeline:

Stage 1: Scenario Interpretation (existing)
Stage 2: Multi-Agent Knowledge Graph Retrieval (new)
Stage 3: Multi-Agent Test Generation with Iterative Refinement (new)

Key Enhancements:
- Multi-hop intelligent KG retrieval with priority-guided expansion
- Planner Agent for test strategy
- Test Architect Agent for structure design
- Test Writer Agent for generation
- Reviewer Agent for quality assessment
- Iterative refinement until quality threshold met

Usage:
    python multi_agent_pipeline.py "User logs in → Views dashboard → Creates todo" --iterations 3
"""

from __future__ import annotations

import os
import json
import sys
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

# Import existing Stage 1
from service import Stage1Service, Stage1Config

# Import new multi-agent systems
from multi_agent_retrieval import (
    multi_agent_retrieve,
    MultiAgentRetrievalConfig,
    RetrievalStrategy
)

from multi_agent_test_generation import (
    multi_agent_generate,
    MultiAgentTestConfig,
    TestFramework,
    OutputFormat
)


@dataclass
class MultiAgentPipelineConfig:
    """Configuration for the complete multi-agent pipeline"""
    
    # Stage 1 config
    stage1_config: Optional[Stage1Config] = None
    
    # Stage 2 config (Multi-Agent Retrieval)
    retrieval_config: Optional[MultiAgentRetrievalConfig] = None
    
    # Stage 3 config (Multi-Agent Test Generation)
    test_gen_config: Optional[MultiAgentTestConfig] = None
    
    # Pipeline settings
    save_intermediate: bool = True
    output_dir: str = "/tmp/multi_agent_pipeline"
    verbose: bool = True


class MultiAgentPipeline:
    """Orchestrator for the complete multi-agent pipeline"""
    
    def __init__(self, config: Optional[MultiAgentPipelineConfig] = None):
        self.config = config or MultiAgentPipelineConfig()
        
        # Initialize Stage 1 service
        self.stage1_service = Stage1Service(self.config.stage1_config)
        
        # Configs for Stage 2 and 3 will be used on-demand
        self.retrieval_config = self.config.retrieval_config or MultiAgentRetrievalConfig()
        self.test_gen_config = self.config.test_gen_config or MultiAgentTestConfig()
        
        # Create output directory
        if self.config.save_intermediate:
            os.makedirs(self.config.output_dir, exist_ok=True)
    
    def run(self, scenario: str) -> Dict[str, Any]:
        """Execute the complete multi-agent pipeline"""
        
        self._print_header()
        print(f"📝 Scenario: {scenario}")
        print(f"{'='*80}\n")
        
        # Stage 1: Scenario Interpretation
        stage1_output = self._run_stage1(scenario)
        
        if self.config.save_intermediate:
            self._save_json(stage1_output, "stage1_output.json")
        
        # Stage 2: Multi-Agent Knowledge Graph Retrieval
        stage2_output = self._run_stage2(stage1_output)
        
        if self.config.save_intermediate:
            self._save_json(stage2_output, "stage2_output.json")
        
        # Stage 3: Multi-Agent Test Generation
        stage3_output = self._run_stage3(stage1_output, stage2_output)
        
        if self.config.save_intermediate:
            self._save_json(stage3_output, "stage3_output.json")
            if stage3_output.get("test_code"):
                self._save_file(stage3_output["test_code"], "generated_test_code.txt")
            if stage3_output.get("test_specs"):
                self._save_json(stage3_output["test_specs"], "generated_test_specs.json")
        
        # Build complete output
        complete_output = {
            "scenario": scenario,
            "pipeline_config": {
                "retrieval_strategy": self.retrieval_config.strategy.value,
                "retrieval_iterations": self.retrieval_config.max_iterations,
                "test_framework": self.test_gen_config.test_framework.value,
                "test_output_format": self.test_gen_config.output_format.value,
                "refinement_iterations": self.test_gen_config.max_refinement_iterations,
                "quality_threshold": self.test_gen_config.quality_threshold
            },
            "stage1": stage1_output,
            "stage2": stage2_output,
            "stage3": stage3_output
        }
        
        # Print summary
        self._print_summary(complete_output)
        
        return complete_output
    
    def _run_stage1(self, scenario: str) -> Dict[str, Any]:
        """Run Stage 1: Scenario Interpretation"""
        
        print(f"\n{'='*80}")
        print("📊 STAGE 1: Scenario Interpretation & Entity Extraction")
        print(f"{'='*80}\n")
        
        output = self.stage1_service.run_stage1(scenario)
        
        print(f"✓ Found {len(output['routes'])} routes")
        if output['routes']:
            for r in output['routes'][:3]:
                print(f"  - {r['name']} (score: {r['score']:.3f})")
        
        print(f"✓ Found {len(output['components'])} components")
        if output['components']:
            for c in output['components'][:3]:
                print(f"  - {c['name']} (score: {c['score']:.3f})")
        
        if 'structured' in output:
            structured = output['structured']
            print(f"\n✓ Structured Interpretation:")
            print(f"  - Type: {structured['type']}")
            print(f"  - Goal: {structured['goal']}")
            print(f"  - Auth Required: {structured['auth_required']}")
            print(f"  - Priority: {structured['priority']}")
        
        return output
    
    def _run_stage2(self, stage1_output: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 2: Multi-Agent Knowledge Graph Retrieval"""
        
        print(f"\n{'='*80}")
        print("🕸️  STAGE 2: Multi-Agent Knowledge Graph Retrieval")
        print(f"{'='*80}")
        print(f"Strategy: {self.retrieval_config.strategy.value}")
        print(f"Max Iterations: {self.retrieval_config.max_iterations}")
        print(f"Expansion Depth: {self.retrieval_config.expansion_depth}")
        print(f"{'='*80}\n")
        
        # Prepare input for multi-agent retrieval
        structured = stage1_output.get('structured')
        if not structured:
            # Fallback: create basic structured input
            structured = {
                "routes": [r['name'] for r in stage1_output.get('routes', [])[:5]],
                "components": [c['name'] for c in stage1_output.get('components', [])[:5]],
                "goal": stage1_output.get('scenario', '')
            }
        
        # Execute multi-agent retrieval
        output = multi_agent_retrieve(structured, self.retrieval_config)
        
        return output
    
    def _run_stage3(
        self,
        stage1_output: Dict[str, Any],
        stage2_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Stage 3: Multi-Agent Test Generation"""
        
        print(f"\n{'='*80}")
        print("🧪 STAGE 3: Multi-Agent Test Generation with Iterative Refinement")
        print(f"{'='*80}")
        print(f"Framework: {self.test_gen_config.test_framework.value}")
        print(f"Output Format: {self.test_gen_config.output_format.value}")
        print(f"Max Refinement Iterations: {self.test_gen_config.max_refinement_iterations}")
        print(f"Quality Threshold: {self.test_gen_config.quality_threshold}")
        print(f"{'='*80}\n")
        
        # Get structured input
        structured = stage1_output.get('structured')
        if not structured:
            structured = {
                "routes": [r['name'] for r in stage1_output.get('routes', [])[:5]],
                "components": [c['name'] for c in stage1_output.get('components', [])[:5]],
                "goal": stage1_output.get('scenario', ''),
                "type": "workflow",
                "auth_required": False,
                "priority": "medium"
            }
        
        # Execute multi-agent test generation
        output = multi_agent_generate(structured, stage2_output, self.test_gen_config)
        
        return output
    
    def _print_header(self):
        """Print pipeline header"""
        print(f"\n{'='*80}")
        print("🚀 MULTI-AGENT TEST GENERATION PIPELINE")
        print("   Enhanced with Intelligent Agents & Iterative Refinement")
        print(f"{'='*80}")
    
    def _print_summary(self, output: Dict[str, Any]):
        """Print pipeline summary"""
        
        print(f"\n{'='*80}")
        print("✅ PIPELINE COMPLETE - SUMMARY")
        print(f"{'='*80}")
        
        # Stage 1 summary
        stage1 = output.get('stage1', {})
        print(f"\n📊 Stage 1:")
        print(f"  - Routes identified: {len(stage1.get('routes', []))}")
        print(f"  - Components identified: {len(stage1.get('components', []))}")
        
        # Stage 2 summary
        stage2 = output.get('stage2', {})
        summary2 = stage2.get('summary', {})
        print(f"\n🕸️  Stage 2 (Multi-Agent Retrieval):")
        print(f"  - Total nodes retrieved: {summary2.get('node_count', 0)}")
        print(f"  - Total edges retrieved: {summary2.get('edge_count', 0)}")
        print(f"  - Retrieval iterations: {summary2.get('iterations', 0)}")
        node_types = summary2.get('node_types', {})
        if node_types:
            top_types = sorted(node_types.items(), key=lambda x: -x[1])[:5]
            print(f"  - Top node types: {', '.join([f'{k}({v})' for k, v in top_types])}")
        
        # Stage 3 summary
        stage3 = output.get('stage3', {})
        metadata3 = stage3.get('metadata', {})
        print(f"\n🧪 Stage 3 (Multi-Agent Test Generation):")
        print(f"  - Refinement iterations: {metadata3.get('refinement_iterations', 0)}")
        print(f"  - Final quality score: {metadata3.get('final_quality_score', 'N/A')}")
        
        if stage3.get('test_plan'):
            test_plan = stage3['test_plan']
            print(f"  - Test scenarios planned: {len(test_plan.get('scenarios', []))}")
            print(f"  - Priority areas: {', '.join(test_plan.get('priority_areas', [])[:3])}")
        
        if stage3.get('test_specs'):
            print(f"  - Test specifications generated: {len(stage3['test_specs'])}")
        
        if stage3.get('test_code'):
            print(f"  - Test code generated: {len(stage3['test_code'])} characters")
        
        if stage3.get('reviews'):
            reviews = stage3['reviews']
            print(f"\n📝 Quality Evolution:")
            for review in reviews:
                it = review.get('iteration', 0)
                score = review.get('score', 0)
                print(f"  - Iteration {it}: {score:.2f}")
        
        if self.config.save_intermediate:
            print(f"\n💾 Output saved to: {self.config.output_dir}/")
        
        print(f"\n{'='*80}\n")
    
    def _save_json(self, data: Any, filename: str):
        """Save JSON data to file"""
        filepath = os.path.join(self.config.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        if self.config.verbose:
            print(f"  💾 Saved: {filepath}")
    
    def _save_file(self, content: str, filename: str):
        """Save text content to file"""
        filepath = os.path.join(self.config.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        if self.config.verbose:
            print(f"  💾 Saved: {filepath}")


def run_multi_agent_pipeline(
    scenario: str,
    **kwargs
) -> Dict[str, Any]:
    """Run the complete multi-agent pipeline
    
    Args:
        scenario: User scenario text
        **kwargs: Configuration options
            - retrieval_iterations: Max retrieval iterations (default: 3)
            - retrieval_strategy: 'breadth_first' or 'priority_guided' (default: priority_guided)
            - test_framework: 'playwright', 'cypress', or 'selenium' (default: playwright)
            - output_format: 'code', 'spec', or 'both' (default: both)
            - refinement_iterations: Max refinement iterations (default: 3)
            - quality_threshold: Quality threshold for stopping (default: 0.8)
            - output_dir: Directory for intermediate outputs (default: /tmp/multi_agent_pipeline)
    
    Returns:
        Complete pipeline output
    """
    
    # Build configuration
    config = MultiAgentPipelineConfig()
    config.output_dir = kwargs.get('output_dir', '/tmp/multi_agent_pipeline')
    config.save_intermediate = kwargs.get('save_intermediate', True)
    config.verbose = kwargs.get('verbose', True)
    
    # Retrieval config
    retrieval_config = MultiAgentRetrievalConfig()
    retrieval_config.max_iterations = kwargs.get('retrieval_iterations', 3)
    retrieval_config.strategy = RetrievalStrategy(kwargs.get('retrieval_strategy', 'priority_guided'))
    config.retrieval_config = retrieval_config
    
    # Test generation config
    test_gen_config = MultiAgentTestConfig()
    test_gen_config.test_framework = TestFramework(kwargs.get('test_framework', 'playwright'))
    test_gen_config.output_format = OutputFormat(kwargs.get('output_format', 'both'))
    test_gen_config.max_refinement_iterations = kwargs.get('refinement_iterations', 3)
    test_gen_config.quality_threshold = kwargs.get('quality_threshold', 0.8)
    config.test_gen_config = test_gen_config
    
    # Run pipeline
    pipeline = MultiAgentPipeline(config)
    return pipeline.run(scenario)


# --------------- CLI ---------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent Test Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python multi_agent_pipeline.py "User logs in and views dashboard"
  
  # With custom iterations
  python multi_agent_pipeline.py "Create and edit todo" --retrieval-iterations 4 --refinement-iterations 3
  
  # With Cypress and specs only
  python multi_agent_pipeline.py "Login flow" --framework cypress --format spec
  
  # Save to specific directory
  python multi_agent_pipeline.py "Register user" --output-dir ./test_outputs
        """
    )
    
    parser.add_argument(
        "scenario",
        type=str,
        help="User flow scenario (e.g., 'Login → Dashboard → Create Todo')"
    )
    
    # Retrieval options
    parser.add_argument(
        "--retrieval-iterations",
        type=int,
        default=3,
        help="Max retrieval iterations (default: 3)"
    )
    parser.add_argument(
        "--retrieval-strategy",
        type=str,
        choices=["breadth_first", "priority_guided"],
        default="priority_guided",
        help="Retrieval strategy (default: priority_guided)"
    )
    
    # Test generation options
    parser.add_argument(
        "--framework",
        type=str,
        choices=["playwright", "cypress", "selenium"],
        default="playwright",
        help="Test framework (default: playwright)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["code", "spec", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--refinement-iterations",
        type=int,
        default=3,
        help="Max refinement iterations (default: 3)"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.8,
        help="Quality threshold for stopping refinement (default: 0.8)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/multi_agent_pipeline",
        help="Output directory for intermediate files (default: /tmp/multi_agent_pipeline)"
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Don't save intermediate outputs"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Save complete output as JSON to this file"
    )
    
    # Agent control
    parser.add_argument(
        "--disable-planner",
        action="store_true",
        help="Disable planner agent"
    )
    parser.add_argument(
        "--disable-reviewer",
        action="store_true",
        help="Disable reviewer agent"
    )
    parser.add_argument(
        "--disable-refinement",
        action="store_true",
        help="Disable iterative refinement"
    )
    
    args = parser.parse_args()
    
    try:
        # Run pipeline
        result = run_multi_agent_pipeline(
            scenario=args.scenario,
            retrieval_iterations=args.retrieval_iterations,
            retrieval_strategy=args.retrieval_strategy,
            test_framework=args.framework,
            output_format=args.format,
            refinement_iterations=args.refinement_iterations,
            quality_threshold=args.quality_threshold,
            output_dir=args.output_dir,
            save_intermediate=not args.no_save_intermediate,
            verbose=True
        )
        
        # Update test gen config based on flags
        if args.disable_planner:
            result['stage3']['test_plan'] = None
        if args.disable_reviewer or args.disable_refinement:
            result['stage3']['reviews'] = []
        
        # Save complete output if requested
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Complete output saved: {args.output_json}")
        
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


