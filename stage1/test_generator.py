"""
Stage 3 — E2E Test Case Generation

Generates end-to-end UI test cases from:
- Structured scenario interpretation (Stage 1B output)
- Knowledge graph subgraph (Stage 2A output)

Supports multiple test frameworks:
- Playwright (default)
- Cypress
- Selenium

Environment variables:
- STAGE3_TEST_FRAMEWORK (default: playwright)
- STAGE3_LLM_MODEL (e.g., groq/llama-3.3-70b-versatile, gpt-4o-mini)
- STAGE3_TEMPERATURE (default: 0.1)
- STAGE3_MAX_TOKENS (default: 2000)
- GROQ_API_KEY or OPENAI_API_KEY (depending on model)

Usage:
>>> from stage1.test_generator import TestGenerator
>>> generator = TestGenerator()
>>> test_code = generator.generate_test(structured_scenario, subgraph)
>>> print(test_code)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

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


class TestFramework(str, Enum):
    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"
    SELENIUM = "selenium"


@dataclass
class Stage3Config:
    """Configuration for test generation."""
    test_framework: TestFramework = TestFramework(os.getenv("STAGE3_TEST_FRAMEWORK", "playwright"))
    llm_model: str = os.getenv("STAGE3_LLM_MODEL", "groq/llama-3.3-70b-versatile")
    temperature: float = float(os.getenv("STAGE3_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("STAGE3_MAX_TOKENS", "2000"))
    include_comments: bool = os.getenv("STAGE3_INCLUDE_COMMENTS", "true").lower() in ("true", "1", "yes")
    include_assertions: bool = os.getenv("STAGE3_INCLUDE_ASSERTIONS", "true").lower() in ("true", "1", "yes")


class TestGenerator:
    """Generates E2E test cases from structured scenarios and KG subgraphs."""
    
    def __init__(self, config: Optional[Stage3Config] = None):
        self.config = config or Stage3Config()
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError(
                "LangChain dependencies not installed. "
                "Install with: pip install langchain-community"
            )
    
    def generate_test(
        self,
        structured: Dict[str, Any],
        subgraph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate E2E test case from structured scenario and subgraph.
        
        Args:
            structured: Output from Stage 1B (scenario interpretation)
            subgraph: Output from Stage 2A (filtered subgraph)
        
        Returns:
            {
                "test_code": str,  # Generated test code
                "test_name": str,  # Test function/describe block name
                "framework": str,  # Test framework used
                "metadata": {
                    "routes": List[str],
                    "components": List[str],
                    "assertions_count": int,
                    "steps_count": int
                }
            }
        """
        # Build context from subgraph
        context = self._build_context_from_subgraph(subgraph)
        
        # Generate test code using LLM
        test_code = self._generate_with_llm(structured, context)
        
        # Extract metadata
        metadata = self._extract_metadata(test_code, structured, subgraph)
        
        return {
            "test_code": test_code,
            "test_name": self._generate_test_name(structured),
            "framework": self.config.test_framework.value,
            "metadata": metadata
        }
    
    def _build_context_from_subgraph(self, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured context from KG subgraph for test generation."""
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        summary = subgraph.get("summary", {})
        
        # Organize nodes by type
        components = []
        routes = []
        states = []
        event_handlers = []
        hooks = []
        props = []
        
        for node in nodes:
            node_type = node.get("label", "")
            name = node.get("name", "")
            node_id = node.get("id", "")
            
            node_info = {
                "id": node_id,
                "name": name,
                "properties": node.get("properties", {})
            }
            
            if node_type == "Component":
                components.append(node_info)
            elif node_type == "Route":
                routes.append(node_info)
            elif node_type == "State":
                states.append(node_info)
            elif node_type == "EventHandler":
                event_handlers.append(node_info)
            elif node_type == "Hook":
                hooks.append(node_info)
            elif node_type == "Prop":
                props.append(node_info)
        
        # Build component -> state/handlers mapping
        component_details = {}
        for comp in components:
            comp_id = comp["id"]
            comp_details = {
                "name": comp["name"],
                "states": [],
                "handlers": [],
                "props": [],
                "hooks": []
            }
            
            # Find connected states, handlers, props, hooks
            for edge in edges:
                if edge.get("source") == comp_id:
                    target_id = edge.get("target")
                    edge_type = edge.get("type", "")
                    
                    # Find target node
                    target_node = next((n for n in nodes if n.get("id") == target_id), None)
                    if target_node:
                        target_label = target_node.get("label", "")
                        target_name = target_node.get("name", "")
                        
                        if target_label == "State":
                            comp_details["states"].append(target_name)
                        elif target_label == "EventHandler":
                            comp_details["handlers"].append(target_name)
                        elif target_label == "Prop":
                            comp_details["props"].append(target_name)
                        elif target_label == "Hook":
                            comp_details["hooks"].append(target_name)
            
            component_details[comp["name"]] = comp_details
        
        return {
            "components": components,
            "routes": routes,
            "states": states,
            "event_handlers": event_handlers,
            "hooks": hooks,
            "props": props,
            "component_details": component_details,
            "summary": summary
        }
    
    def _generate_with_llm(
        self,
        structured: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate test code using LLM."""
        
        # Build the prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(structured, context)
        
        try:
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
            
            # Extract code block if wrapped in markdown
            code = self._extract_code_block(content)
            
            return code
            
        except Exception as e:
            import sys
            print(f"[Stage 3] Test generation failed: {type(e).__name__}: {e}", file=sys.stderr)
            # Return a template as fallback
            return self._generate_fallback_test(structured, context)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for test generation."""
        framework = self.config.test_framework.value
        
        if framework == "playwright":
            framework_docs = """
## Playwright Framework Guidelines

1. **Test Structure**:
   ```typescript
   import { test, expect } from '@playwright/test';
   
   test.describe('Feature Name', () => {
     test('should perform action', async ({ page }) => {
       // test code
     });
   });
   ```

2. **Navigation**: `await page.goto('/route')`
3. **Selectors**: Use data-testid when available, fallback to role/text
   - `page.getByTestId('login-button')`
   - `page.getByRole('button', { name: 'Login' })`
   - `page.getByText('Welcome')`
4. **Actions**:
   - Click: `await page.getByRole('button').click()`
   - Fill: `await page.getByLabel('Email').fill('user@example.com')`
   - Select: `await page.selectOption('select', 'value')`
5. **Assertions**:
   - `await expect(page).toHaveURL('/dashboard')`
   - `await expect(page.getByText('Success')).toBeVisible()`
   - `await expect(page.locator('.error')).toHaveCount(0)`
6. **Waiting**: `await page.waitForURL('/dashboard')` or `await page.waitForSelector('.loaded')`
"""
        elif framework == "cypress":
            framework_docs = """
## Cypress Framework Guidelines

1. **Test Structure**:
   ```javascript
   describe('Feature Name', () => {
     it('should perform action', () => {
       // test code
     });
   });
   ```

2. **Navigation**: `cy.visit('/route')`
3. **Selectors**: Use data-cy when available, fallback to contains
   - `cy.get('[data-cy="login-button"]')`
   - `cy.contains('button', 'Login')`
4. **Actions**:
   - Click: `cy.get('button').click()`
   - Type: `cy.get('input[name="email"]').type('user@example.com')`
   - Select: `cy.get('select').select('value')`
5. **Assertions**:
   - `cy.url().should('include', '/dashboard')`
   - `cy.contains('Success').should('be.visible')`
   - `cy.get('.error').should('not.exist')`
6. **Waiting**: `cy.wait('@apiCall')` or `cy.get('.loaded').should('exist')`
"""
        else:  # selenium
            framework_docs = """
## Selenium Framework Guidelines

1. **Test Structure**:
   ```python
   import unittest
   from selenium import webdriver
   from selenium.webdriver.common.by import By
   
   class TestFeature(unittest.TestCase):
       def setUp(self):
           self.driver = webdriver.Chrome()
       
       def test_action(self):
           # test code
       
       def tearDown(self):
           self.driver.quit()
   ```

2. **Navigation**: `driver.get('http://localhost:3000/route')`
3. **Selectors**: Use data-testid or CSS/XPath
   - `driver.find_element(By.CSS_SELECTOR, '[data-testid="login-button"]')`
   - `driver.find_element(By.XPATH, '//button[text()="Login"]')`
4. **Actions**:
   - Click: `element.click()`
   - Type: `element.send_keys('user@example.com')`
   - Select: `Select(element).select_by_value('value')`
5. **Assertions**:
   - `self.assertIn('/dashboard', driver.current_url)`
   - `self.assertTrue(element.is_displayed())`
6. **Waiting**: Use WebDriverWait with expected_conditions
"""
        
        return f"""You are an expert E2E test engineer specializing in {framework} tests for React applications.

Your task is to generate production-ready, maintainable E2E test cases based on:
1. A structured user flow scenario
2. Knowledge graph context (components, routes, states, event handlers)

{framework_docs}

## Test Generation Rules

1. **Completeness**: Include all necessary steps from start to finish
2. **Robustness**: Add proper waits and error handling
3. **Maintainability**: Use clear selectors and descriptive names
4. **Assertions**: Verify state changes, navigation, and UI feedback
5. **Comments**: Add inline comments explaining complex interactions
6. **Best Practices**: Follow framework-specific patterns

## Output Format

Generate ONLY the test code. Do not include explanations before or after.
The code should be production-ready and executable.
Include all necessary imports and setup.
"""
    
    def _build_user_prompt(
        self,
        structured: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build the user prompt with scenario and context."""
        
        # Extract scenario details
        scenario_type = structured.get("type", "workflow")
        routes = structured.get("routes", [])
        components = structured.get("components", [])
        auth_required = structured.get("auth_required", False)
        goal = structured.get("goal", "")
        
        # Build component details section
        comp_details_text = []
        for comp_name, details in context["component_details"].items():
            if comp_name in components:
                parts = [f"  - **{comp_name}**:"]
                if details["states"]:
                    parts.append(f"    - States: {', '.join(details['states'])}")
                if details["handlers"]:
                    parts.append(f"    - Event Handlers: {', '.join(details['handlers'])}")
                if details["props"]:
                    parts.append(f"    - Props: {', '.join(details['props'][:5])}")
                comp_details_text.append("\n".join(parts))
        
        prompt = f"""Generate an E2E test for the following scenario:

## Scenario Details
- **Type**: {scenario_type}
- **Goal**: {goal}
- **Authentication Required**: {auth_required}

## User Flow
Routes (in order): {' → '.join(routes) if routes else 'N/A'}

## Components Involved
{chr(10).join(comp_details_text) if comp_details_text else '  No component details available'}

## Available Routes
{json.dumps([r['name'] for r in context['routes']], indent=2)}

## Test Requirements

1. **Setup**: {'Include login flow before main test steps' if auth_required else 'No authentication required'}
2. **Navigation**: Follow the route sequence: {' → '.join(routes)}
3. **Interactions**: Test user interactions with the components
4. **State Verification**: Verify state changes where applicable
5. **Success Criteria**: Confirm successful completion of the workflow

## Additional Context
- Total components in context: {len(context['components'])}
- Total event handlers: {len(context['event_handlers'])}
- Total states: {len(context['states'])}

Generate a complete, executable test case following the framework guidelines.
"""
        
        return prompt
    
    def _extract_code_block(self, content: str) -> str:
        """Extract code from markdown code blocks."""
        import re
        
        # Try to find code block with language
        pattern = r"```(?:typescript|javascript|python|ts|js|py)\n(.*?)\n```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try to find any code block
        pattern = r"```\n(.*?)\n```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # No code block found, return as-is
        return content.strip()
    
    def _generate_fallback_test(
        self,
        structured: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a basic template test as fallback."""
        framework = self.config.test_framework.value
        routes = structured.get("routes", [])
        goal = structured.get("goal", "test scenario")
        
        if framework == "playwright":
            return f"""import {{ test, expect }} from '@playwright/test';

test.describe('{goal}', () => {{
  test('should complete user flow', async ({{ page }}) => {{
    // Navigate through routes
{chr(10).join([f"    await page.goto('{route}');" for route in routes])}
    
    // Add assertions
    await expect(page).toHaveURL('{routes[-1] if routes else '/'}');
  }});
}});
"""
        elif framework == "cypress":
            return f"""describe('{goal}', () => {{
  it('should complete user flow', () => {{
    // Navigate through routes
{chr(10).join([f"    cy.visit('{route}');" for route in routes])}
    
    // Add assertions
    cy.url().should('include', '{routes[-1] if routes else '/'}');
  }});
}});
"""
        else:  # selenium
            return f"""import unittest
from selenium import webdriver

class TestScenario(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
    
    def test_user_flow(self):
        '''Test: {goal}'''
        # Navigate through routes
{chr(10).join([f"        self.driver.get('http://localhost:3000{route}')" for route in routes])}
        
        # Add assertions
        self.assertIn('{routes[-1] if routes else '/'}', self.driver.current_url)
    
    def tearDown(self):
        self.driver.quit()
"""
    
    def _generate_test_name(self, structured: Dict[str, Any]) -> str:
        """Generate a test name from structured scenario."""
        goal = structured.get("goal", "")
        scenario_type = structured.get("type", "workflow")
        
        # Convert goal to snake_case
        import re
        name = re.sub(r'[^\w\s-]', '', goal.lower())
        name = re.sub(r'[-\s]+', '_', name)
        
        return f"test_{scenario_type}_{name}"[:80]
    
    def _extract_metadata(
        self,
        test_code: str,
        structured: Dict[str, Any],
        subgraph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata from generated test."""
        import re
        
        # Count test steps (rough estimate)
        steps = len(re.findall(r'(await|cy\.|driver\.)', test_code))
        
        # Count assertions
        assertions = len(re.findall(r'(expect|assert|should)', test_code, re.IGNORECASE))
        
        return {
            "routes": structured.get("routes", []),
            "components": structured.get("components", []),
            "assertions_count": assertions,
            "steps_count": steps,
            "auth_required": structured.get("auth_required", False),
            "priority": structured.get("priority", "medium"),
            "nodes_analyzed": subgraph.get("summary", {}).get("node_count", 0),
            "edges_analyzed": subgraph.get("summary", {}).get("edge_count", 0)
        }


# --------------- Public API ---------------
def generate_test(
    structured: Dict[str, Any],
    subgraph: Dict[str, Any],
    config: Optional[Stage3Config] = None
) -> Dict[str, Any]:
    """Generate E2E test case from structured scenario and subgraph.
    
    Args:
        structured: Output from Stage 1B
        subgraph: Output from Stage 2A
        config: Optional Stage3Config instance
    
    Returns:
        {
            "test_code": str,
            "test_name": str,
            "framework": str,
            "metadata": dict
        }
    """
    generator = TestGenerator(config)
    return generator.generate_test(structured, subgraph)


# --------------- CLI ---------------
def _main_cli():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Stage 3 — E2E Test Case Generation")
    parser.add_argument("--input", type=str, required=True, help="JSON file with Stage 1+2A output")
    parser.add_argument("--output", type=str, help="Output file for generated test (default: stdout)")
    parser.add_argument("--framework", type=str, choices=["playwright", "cypress", "selenium"],
                       default="playwright", help="Test framework (default: playwright)")
    parser.add_argument("--model", type=str, help="LLM model override")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Extract structured and subgraph
    if "stage1" in data and "stage2a" in data:
        # Pipeline output format
        structured = data["stage1"].get("structured")
        subgraph = data["stage2a"]
    elif "structured" in data and "nodes" in data:
        # Combined format
        structured = data["structured"]
        subgraph = data
    else:
        print("error: Input file must contain structured scenario and subgraph", file=sys.stderr)
        sys.exit(1)
    
    if not structured:
        print("error: No structured scenario found in input", file=sys.stderr)
        sys.exit(1)
    
    # Configure
    config = Stage3Config()
    if args.framework:
        config.test_framework = TestFramework(args.framework)
    if args.model:
        config.llm_model = args.model
    
    # Generate test
    result = generate_test(structured, subgraph, config)
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(result["test_code"])
        
        # Also write metadata
        metadata_file = args.output.rsplit(".", 1)[0] + "_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({
                "test_name": result["test_name"],
                "framework": result["framework"],
                "metadata": result["metadata"]
            }, f, indent=2)
        
        print(f"Test generated: {args.output}", file=sys.stderr)
        print(f"Metadata saved: {metadata_file}", file=sys.stderr)
    else:
        print(result["test_code"])


if __name__ == "__main__":
    _main_cli()

