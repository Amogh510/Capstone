"""
Stage 3 - Test Case Specification Generator

Generates structured test case specifications in JSON format similar to traditional test cases.
Output includes: ID, Title, Steps, Expected Results

This is an alternative to executable test code generation for documentation and planning.

Usage:
>>> from stage1.test_spec_generator import generate_test_specs
>>> specs = generate_test_specs(structured_scenario, subgraph)
>>> print(json.dumps(specs, indent=2))
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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


@dataclass
class TestSpecConfig:
    """Configuration for test specification generation."""
    llm_model: str = os.getenv("STAGE3_LLM_MODEL", "groq/llama-3.3-70b-versatile")
    temperature: float = float(os.getenv("STAGE3_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("STAGE3_MAX_TOKENS", "6000"))
    num_test_cases: int = int(os.getenv("STAGE3_NUM_TEST_CASES", "5"))


class TestSpecGenerator:
    """Generates structured test case specifications in JSON format."""
    
    def __init__(self, config: Optional[TestSpecConfig] = None):
        self.config = config or TestSpecConfig()
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError(
                "LangChain dependencies not installed. "
                "Install with: pip install langchain-community"
            )
    
    def generate_specs(
        self,
        structured: Dict[str, Any],
        subgraph: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate test case specifications from structured scenario and subgraph.
        
        Args:
            structured: Output from Stage 1B (scenario interpretation)
            subgraph: Output from Stage 2A (filtered subgraph)
        
        Returns:
            List of test case specifications in format:
            [
                {
                    "id": "TC01",
                    "title": "...",
                    "steps": ["...", "..."],
                    "expected_result": "..."
                }
            ]
        """
        # Build context from subgraph
        context = self._build_context_from_subgraph(subgraph)
        
        # Generate test specifications using LLM
        try:
            test_specs = self._generate_with_llm(structured, context)
            return test_specs
        except Exception as e:
            print(f"[Test Spec Generator] LLM generation failed: {e}")
            # Fallback to basic template
            return self._generate_fallback_specs(structured, context)
    
    def _build_context_from_subgraph(self, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured context from KG subgraph."""
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        
        # Organize nodes by type
        components = {}
        routes = []
        states = []
        event_handlers = []
        
        for node in nodes:
            node_type = node.get("label", "")
            name = node.get("name", "")
            node_id = node.get("id", "")
            
            if node_type == "Component":
                components[name] = {"id": node_id, "states": [], "handlers": [], "props": []}
            elif node_type == "Route":
                routes.append(name)
            elif node_type == "State":
                states.append(name)
            elif node_type == "EventHandler":
                event_handlers.append(name)
        
        # Build component relationships
        for edge in edges:
            source_node = next((n for n in nodes if n.get("id") == edge.get("source")), None)
            target_node = next((n for n in nodes if n.get("id") == edge.get("target")), None)
            
            if source_node and target_node:
                source_name = source_node.get("name", "")
                target_name = target_node.get("name", "")
                target_type = target_node.get("label", "")
                
                if source_name in components:
                    if target_type == "State":
                        components[source_name]["states"].append(target_name)
                    elif target_type == "EventHandler":
                        components[source_name]["handlers"].append(target_name)
                    elif target_type == "Prop":
                        components[source_name]["props"].append(target_name)
        
        return {
            "components": components,
            "routes": routes,
            "states": states,
            "event_handlers": event_handlers,
            "summary": subgraph.get("summary", {})
        }
    
    def _generate_with_llm(
        self,
        structured: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate test specifications using LLM."""
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(structured, context)
        
        llm = ChatLiteLLM(
            model=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        content = getattr(response, "content", "")
        
        # Extract JSON from response
        test_specs = self._extract_json_array(content)
        
        return test_specs
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for test specification generation."""
        return """You are an expert QA engineer creating comprehensive E2E test case specifications.

Generate test cases in JSON array format with this exact structure:
[
  {
    "id": "TC01",
    "title": "Descriptive test case title",
    "steps": [
      "Step 1: Action to perform",
      "Step 2: Another action",
      "Step 3: Verify something"
    ],
    "expected_result": "Clear description of expected outcome",
    "rationale": "Explanation of why this test was generated based on KG context (components, states, handlers, routes used)"
  }
]

## Test Case Guidelines

1. **Coverage**: Include happy path, error cases, edge cases, and accessibility
2. **Specificity**: Reference actual routes, components, and UI elements from the context
3. **Completeness**: Each test should be self-contained and executable
4. **Clarity**: Steps should be clear enough for manual or automated testing
5. **Realistic**: Use realistic test data (emails, names, etc.)
6. **Explainability**: The rationale field MUST explain:
   - Which components from the KG are being tested
   - Which routes are involved
   - Which states/handlers are being verified
   - Why this test case is important for the scenario
   - What KG context informed this test design

## Test Case Types to Consider

- **Happy Path**: Main user flow works correctly
- **Authentication**: Login success/failure, session handling
- **Data Operations**: CRUD operations on entities
- **Error Handling**: API failures, validation errors, network issues
- **State Management**: Component state updates correctly
- **Navigation**: Route transitions work properly
- **Accessibility**: Keyboard navigation, ARIA attributes, focus management
- **Edge Cases**: Empty states, long text, special characters

## Output Requirements

- Generate 5-10 test cases covering different scenarios
- Use sequential IDs: TC01, TC02, TC03, etc.
- Include pre-requisites in steps (e.g., "Login successfully")
- Be specific about selectors, text, and UI elements
- Output ONLY the JSON array, no markdown formatting
"""
    
    def _build_user_prompt(
        self,
        structured: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build user prompt with scenario and context."""
        
        scenario_type = structured.get("type", "workflow")
        routes = structured.get("routes", [])
        components_list = structured.get("components", [])
        auth_required = structured.get("auth_required", False)
        goal = structured.get("goal", "")
        priority = structured.get("priority", "medium")
        
        # Build component details
        comp_details = []
        for comp_name in components_list:
            if comp_name in context["components"]:
                details = context["components"][comp_name]
                comp_details.append(f"- {comp_name}:")
                if details["states"]:
                    comp_details.append(f"  - States: {', '.join(details['states'][:3])}")
                if details["handlers"]:
                    comp_details.append(f"  - Handlers: {', '.join(details['handlers'][:3])}")
                if details["props"]:
                    comp_details.append(f"  - Props: {', '.join(details['props'][:5])}")
        
        prompt = f"""Generate comprehensive E2E test case specifications for this scenario:

## Scenario Information
- **Type**: {scenario_type}
- **Goal**: {goal}
- **Priority**: {priority}
- **Authentication Required**: {auth_required}

## User Flow
Routes: {' → '.join(routes) if routes else 'N/A'}

## Available Components
{chr(10).join(comp_details) if comp_details else 'No detailed component information'}

## Available Routes
{', '.join(context['routes'])}

## Event Handlers Available
{', '.join(context['event_handlers'][:10]) if context['event_handlers'] else 'None found'}

## Requirements

Generate {self.config.num_test_cases} test cases that cover:
1. Primary happy path for the workflow
2. Authentication flow (login success/failure)
3. Error scenarios (API failures, validation)
4. Edge cases (empty states, missing data)
5. Accessibility checks (keyboard navigation, ARIA)

Use the actual component names, routes, and handlers from the context above.
Output ONLY a valid JSON array matching the specified format.
"""
        
        return prompt
    
    def _extract_json_array(self, content: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response."""
        import re
        
        # Try to find JSON array
        # First, try to find ```json code block
        pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON array
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        
        # If all else fails, return empty list
        return []
    
    def _generate_fallback_specs(
        self,
        structured: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate basic test specifications as fallback."""
        routes = structured.get("routes", [])
        goal = structured.get("goal", "test scenario")
        auth_required = structured.get("auth_required", False)
        
        specs = []
        
        # TC01: Happy path
        steps = []
        if auth_required:
            steps.extend([
                "Navigate to /login",
                "Enter valid credentials",
                "Click 'Login' button"
            ])
        
        for i, route in enumerate(routes):
            steps.append(f"Navigate to {route}")
            if i < len(routes) - 1:
                steps.append(f"Verify page loaded successfully")
        
        steps.append(f"Verify final URL is {routes[-1] if routes else '/'}")
        
        specs.append({
            "id": "TC01",
            "title": f"Verify {goal} - happy path",
            "steps": steps,
            "expected_result": "User successfully completes the workflow without errors",
            "rationale": f"Tests the primary happy path flow through routes: {', '.join(routes)}. " + 
                        (f"Includes authentication flow as auth is required." if auth_required else "No authentication required.")
        })
        
        # TC02: Authentication failure (if auth required)
        if auth_required:
            specs.append({
                "id": "TC02",
                "title": "Verify invalid login shows error",
                "steps": [
                    "Navigate to /login",
                    "Enter invalid credentials",
                    "Click 'Login' button",
                    "Verify error message is displayed"
                ],
                "expected_result": "Error message shown and user remains on login page",
                "rationale": "Tests error handling in authentication flow. Important because the scenario requires authentication to access protected routes."
            })
        
        return specs


# --------------- Public API ---------------
def generate_test_specs(
    structured: Dict[str, Any],
    subgraph: Dict[str, Any],
    config: Optional[TestSpecConfig] = None
) -> List[Dict[str, Any]]:
    """Generate test case specifications from structured scenario and subgraph.
    
    Args:
        structured: Output from Stage 1B
        subgraph: Output from Stage 2A
        config: Optional TestSpecConfig instance
    
    Returns:
        List of test case specifications
    """
    generator = TestSpecGenerator(config)
    return generator.generate_specs(structured, subgraph)


# --------------- CLI ---------------
def _main_cli():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Stage 3 — Test Case Specification Generator")
    parser.add_argument("--input", type=str, required=True, help="JSON file with Stage 1+2A output")
    parser.add_argument("--output", type=str, help="Output JSON file for test specifications")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of test cases to generate")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Extract structured and subgraph
    if "stage1" in data and "stage2a" in data:
        structured = data["stage1"].get("structured")
        subgraph = data["stage2a"]
    elif "structured" in data and "nodes" in data:
        structured = data["structured"]
        subgraph = data
    else:
        print("error: Input file must contain structured scenario and subgraph", file=sys.stderr)
        sys.exit(1)
    
    if not structured:
        print("error: No structured scenario found in input", file=sys.stderr)
        sys.exit(1)
    
    # Configure
    config = TestSpecConfig()
    if args.num_cases:
        config.num_test_cases = args.num_cases
    
    # Generate test specifications
    specs = generate_test_specs(structured, subgraph, config)
    
    # Output
    output_json = json.dumps(specs, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Test specifications saved: {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    _main_cli()

