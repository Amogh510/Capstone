"""
Multi-Agent Test Generation System with Iterative Refinement

Implements a sophisticated agentic workflow for generating high-quality E2E tests:

Agents:
1. PlannerAgent: Analyzes scenario and creates comprehensive test plan
2. TestArchitectAgent: Designs test structure and identifies test scenarios
3. TestWriterAgent: Generates actual test code/specs
4. ReviewerAgent: Reviews and critiques generated tests
5. RefinementAgent: Iteratively improves tests based on feedback

Workflow:
- Planner creates test strategy
- Architect designs test structure  
- Writer generates initial tests
- Reviewer provides feedback
- Refinement Agent improves tests (2-3 iterations)
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
    """Supported test frameworks"""
    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"
    SELENIUM = "selenium"


class OutputFormat(str, Enum):
    """Output format for test generation"""
    CODE = "code"  # Executable test code
    SPEC = "spec"  # JSON test specifications
    BOTH = "both"  # Both code and specs


@dataclass
class MultiAgentTestConfig:
    """Configuration for multi-agent test generation"""
    
    # LLM settings
    llm_model: str = os.getenv("MULTI_AGENT_TEST_LLM", "groq/llama-3.3-70b-versatile")
    temperature: float = float(os.getenv("MULTI_AGENT_TEST_TEMP", "0.2"))
    max_tokens: int = int(os.getenv("MULTI_AGENT_TEST_MAX_TOKENS", "4000"))
    
    # Test settings
    test_framework: TestFramework = TestFramework(os.getenv("TEST_FRAMEWORK", "playwright"))
    output_format: OutputFormat = OutputFormat(os.getenv("TEST_OUTPUT_FORMAT", "both"))
    num_test_cases: int = int(os.getenv("NUM_TEST_CASES", "5"))
    
    # Iteration settings
    max_refinement_iterations: int = int(os.getenv("MAX_REFINEMENT_ITERATIONS", "3"))
    quality_threshold: float = float(os.getenv("QUALITY_THRESHOLD", "0.8"))
    
    # Agent behavior
    enable_planner: bool = os.getenv("ENABLE_PLANNER", "true").lower() in ("true", "1", "yes")
    enable_reviewer: bool = os.getenv("ENABLE_REVIEWER", "true").lower() in ("true", "1", "yes")
    enable_refinement: bool = os.getenv("ENABLE_REFINEMENT", "true").lower() in ("true", "1", "yes")


@dataclass
class TestPlan:
    """Test plan created by PlannerAgent"""
    test_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    priority_areas: List[str] = field(default_factory=list)
    coverage_goals: List[str] = field(default_factory=list)
    testing_strategy: str = ""
    estimated_complexity: str = "medium"


@dataclass
class TestArchitecture:
    """Test architecture designed by TestArchitectAgent"""
    test_suites: List[Dict[str, Any]] = field(default_factory=list)
    shared_fixtures: List[str] = field(default_factory=list)
    test_flow: str = ""
    assertions_strategy: str = ""


@dataclass
class TestReview:
    """Review feedback from ReviewerAgent"""
    overall_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    missing_coverage: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)


@dataclass
class GenerationContext:
    """Context maintained across the generation workflow"""
    scenario: str
    goal: str
    structured_input: Dict[str, Any]
    subgraph: Dict[str, Any]
    
    # Agent outputs
    test_plan: Optional[TestPlan] = None
    test_architecture: Optional[TestArchitecture] = None
    test_code: Optional[str] = None
    test_specs: Optional[List[Dict[str, Any]]] = None
    
    # Iteration tracking
    reviews: List[TestReview] = field(default_factory=list)
    refinement_iteration: int = 0
    
    # History
    generation_history: List[Dict[str, Any]] = field(default_factory=list)


class PlannerAgent:
    """Agent that creates comprehensive test plan"""
    
    def __init__(self, config: MultiAgentTestConfig):
        self.config = config
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError("LangChain dependencies required")
    
    def create_test_plan(self, ctx: GenerationContext) -> TestPlan:
        """Create comprehensive test plan from scenario and KG context"""
        
        print(f"\n[PlannerAgent] Creating test plan...")
        
        try:
            llm = ChatLiteLLM(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=1500
            )
            
            system_prompt = """You are an expert Test Planner for E2E testing.
Your job is to analyze a user scenario and knowledge graph context to create a comprehensive test plan.

Output a JSON object with this structure:
{
  "test_scenarios": [
    {
      "id": "TS01",
      "name": "Happy path login and navigation",
      "priority": "high",
      "description": "...",
      "coverage_type": "happy_path | error | edge_case | accessibility"
    }
  ],
  "priority_areas": ["Authentication", "Navigation", "Data manipulation"],
  "coverage_goals": ["Verify login flow", "Test error handling", "Check accessibility"],
  "testing_strategy": "Focus on critical user paths first, then error scenarios...",
  "estimated_complexity": "low | medium | high"
}

Consider:
1. User flow steps and goals
2. Critical components and their states
3. Authentication requirements
4. Error scenarios and edge cases
5. Accessibility and UX considerations
"""
            
            # Build context summary
            node_types = ctx.subgraph.get("summary", {}).get("node_types", {})
            routes = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "Route"]
            components = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "Component"]
            states = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "State"]
            handlers = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "EventHandler"]
            
            user_prompt = f"""Scenario: {ctx.scenario}
Goal: {ctx.goal}

Knowledge Graph Context:
- Routes: {', '.join(routes[:10])}
- Components: {', '.join(components[:10])}
- States: {', '.join(states[:8])}
- Event Handlers: {', '.join(handlers[:8])}
- Total Nodes: {ctx.subgraph.get('summary', {}).get('node_count', 0)}

Structured Input:
{json.dumps(ctx.structured_input, indent=2)}

Create a comprehensive test plan:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            
            # Extract JSON
            plan_data = self._extract_json(content)
            
            if plan_data:
                plan = TestPlan(
                    test_scenarios=plan_data.get("test_scenarios", []),
                    priority_areas=plan_data.get("priority_areas", []),
                    coverage_goals=plan_data.get("coverage_goals", []),
                    testing_strategy=plan_data.get("testing_strategy", ""),
                    estimated_complexity=plan_data.get("estimated_complexity", "medium")
                )
                
                print(f"[PlannerAgent] Created plan with {len(plan.test_scenarios)} test scenarios")
                print(f"[PlannerAgent] Priority areas: {', '.join(plan.priority_areas[:5])}")
                
                return plan
            
        except Exception as e:
            print(f"[PlannerAgent] Failed: {e}")
        
        # Fallback plan
        return self._create_fallback_plan(ctx)
    
    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        import re
        
        # Try to find JSON code block
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _create_fallback_plan(self, ctx: GenerationContext) -> TestPlan:
        """Create basic fallback plan"""
        scenarios = [
            {
                "id": "TS01",
                "name": "Happy path flow",
                "priority": "high",
                "description": f"Test main user flow: {ctx.goal}",
                "coverage_type": "happy_path"
            },
            {
                "id": "TS02",
                "name": "Error handling",
                "priority": "medium",
                "description": "Test error scenarios and validation",
                "coverage_type": "error"
            }
        ]
        
        return TestPlan(
            test_scenarios=scenarios,
            priority_areas=["Core functionality"],
            coverage_goals=["Verify main flow works"],
            testing_strategy="Focus on happy path and basic error handling",
            estimated_complexity="medium"
        )


class TestArchitectAgent:
    """Agent that designs test structure"""
    
    def __init__(self, config: MultiAgentTestConfig):
        self.config = config
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError("LangChain dependencies required")
    
    def design_architecture(self, ctx: GenerationContext) -> TestArchitecture:
        """Design test architecture based on plan"""
        
        print(f"\n[TestArchitectAgent] Designing test architecture...")
        
        try:
            llm = ChatLiteLLM(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=1500
            )
            
            system_prompt = f"""You are an expert Test Architect specializing in {self.config.test_framework.value} tests.
Your job is to design the test structure and architecture.

Output a JSON object with this structure:
{{
  "test_suites": [
    {{
      "name": "Authentication Tests",
      "description": "...",
      "test_cases": ["Login success", "Login failure", "Session management"]
    }}
  ],
  "shared_fixtures": ["beforeEach: login", "afterEach: cleanup"],
  "test_flow": "Describe the overall test flow and dependencies...",
  "assertions_strategy": "Describe what to assert and how..."
}}

Consider:
1. Test organization and grouping
2. Shared setup/teardown needs
3. Test dependencies and ordering
4. Assertion strategy (what to verify at each step)
5. Framework-specific best practices for {self.config.test_framework.value}
"""
            
            # Include test plan if available
            plan_summary = ""
            if ctx.test_plan:
                plan_summary = f"""
Test Plan Summary:
- Scenarios: {len(ctx.test_plan.test_scenarios)}
- Priority Areas: {', '.join(ctx.test_plan.priority_areas)}
- Strategy: {ctx.test_plan.testing_strategy[:200]}
"""
            
            user_prompt = f"""Scenario: {ctx.scenario}
{plan_summary}

Framework: {self.config.test_framework.value}
Number of test cases to generate: {self.config.num_test_cases}

Design the test architecture:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            
            # Extract JSON
            arch_data = self._extract_json(content)
            
            if arch_data:
                architecture = TestArchitecture(
                    test_suites=arch_data.get("test_suites", []),
                    shared_fixtures=arch_data.get("shared_fixtures", []),
                    test_flow=arch_data.get("test_flow", ""),
                    assertions_strategy=arch_data.get("assertions_strategy", "")
                )
                
                print(f"[TestArchitectAgent] Designed {len(architecture.test_suites)} test suites")
                
                return architecture
            
        except Exception as e:
            print(f"[TestArchitectAgent] Failed: {e}")
        
        return self._create_fallback_architecture(ctx)
    
    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        import re
        
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _create_fallback_architecture(self, ctx: GenerationContext) -> TestArchitecture:
        """Create basic fallback architecture"""
        return TestArchitecture(
            test_suites=[{
                "name": "Main Flow Tests",
                "description": "Core user flow tests",
                "test_cases": ["Happy path", "Error scenarios"]
            }],
            shared_fixtures=["beforeEach: setup", "afterEach: cleanup"],
            test_flow="Sequential execution of user flow steps",
            assertions_strategy="Verify state and UI changes at each step"
        )


class TestWriterAgent:
    """Agent that generates actual test code/specs"""
    
    def __init__(self, config: MultiAgentTestConfig):
        self.config = config
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError("LangChain dependencies required")
    
    def generate_tests(self, ctx: GenerationContext) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """Generate test code and/or specs based on architecture"""
        
        print(f"\n[TestWriterAgent] Generating tests (format: {self.config.output_format.value})...")
        
        test_code = None
        test_specs = None
        
        if self.config.output_format in [OutputFormat.CODE, OutputFormat.BOTH]:
            test_code = self._generate_test_code(ctx)
        
        if self.config.output_format in [OutputFormat.SPEC, OutputFormat.BOTH]:
            test_specs = self._generate_test_specs(ctx)
        
        return test_code, test_specs
    
    def _generate_test_code(self, ctx: GenerationContext) -> Optional[str]:
        """Generate executable test code"""
        
        try:
            llm = ChatLiteLLM(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            framework = self.config.test_framework.value
            
            system_prompt = f"""You are an expert E2E test engineer specializing in {framework} tests.
Generate production-ready, comprehensive test code based on the test plan and architecture.

Requirements:
1. Follow {framework} best practices and idioms
2. Include proper setup and teardown
3. Add descriptive comments
4. Use appropriate selectors (data-testid preferred)
5. Include comprehensive assertions
6. Handle waits and async operations properly
7. Test both happy path and error scenarios

Output ONLY the test code, no explanations before or after.
"""
            
            # Build comprehensive context
            routes = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "Route"]
            components = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "Component"]
            
            plan_summary = ""
            if ctx.test_plan:
                scenarios_text = "\n".join([
                    f"- {s['id']}: {s['name']} ({s['priority']}) - {s.get('description', '')[:80]}"
                    for s in ctx.test_plan.test_scenarios[:5]
                ])
                plan_summary = f"""
Test Plan:
{scenarios_text}

Strategy: {ctx.test_plan.testing_strategy[:200]}
"""
            
            arch_summary = ""
            if ctx.test_architecture:
                suites_text = "\n".join([
                    f"- {s['name']}: {', '.join(s.get('test_cases', [])[:3])}"
                    for s in ctx.test_architecture.test_suites
                ])
                arch_summary = f"""
Test Architecture:
{suites_text}

Fixtures: {', '.join(ctx.test_architecture.shared_fixtures[:3])}
Assertions Strategy: {ctx.test_architecture.assertions_strategy[:150]}
"""
            
            refinement_feedback = ""
            if ctx.reviews:
                last_review = ctx.reviews[-1]
                refinement_feedback = f"""
Previous Review Feedback (Iteration {ctx.refinement_iteration}):
Score: {last_review.overall_score:.2f}
Weaknesses: {', '.join(last_review.weaknesses[:3])}
Suggestions: {', '.join(last_review.suggestions[:3])}
Missing: {', '.join(last_review.missing_coverage[:2])}

IMPORTANT: Address these issues in the new version!
"""
            
            user_prompt = f"""Scenario: {ctx.scenario}
Goal: {ctx.goal}

Routes: {', '.join(routes[:8])}
Components: {', '.join(components[:8])}
{plan_summary}
{arch_summary}
{refinement_feedback}

Generate complete {framework} test code:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            
            # Extract code block
            code = self._extract_code_block(content)
            
            print(f"[TestWriterAgent] Generated test code ({len(code)} chars)")
            
            return code
            
        except Exception as e:
            print(f"[TestWriterAgent] Test code generation failed: {e}")
            return None
    
    def _generate_test_specs(self, ctx: GenerationContext) -> Optional[List[Dict[str, Any]]]:
        """Generate test specifications in JSON format"""
        
        try:
            llm = ChatLiteLLM(
                model=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            system_prompt = """You are an expert QA engineer creating comprehensive E2E test specifications.

Generate test cases in JSON array format:
[
  {
    "id": "TC01",
    "title": "Descriptive test case title",
    "priority": "high | medium | low",
    "steps": [
      "Step 1: Action to perform",
      "Step 2: Another action"
    ],
    "expected_result": "Clear description of expected outcome",
    "rationale": "Why this test is important, what it covers from the KG",
    "coverage_type": "happy_path | error | edge_case | accessibility"
  }
]

Generate 5-10 diverse test cases covering different scenarios.
Output ONLY the JSON array, no markdown formatting.
"""
            
            routes = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "Route"]
            components = [n["name"] for n in ctx.subgraph.get("nodes", []) if n.get("label") == "Component"]
            
            plan_summary = ""
            if ctx.test_plan:
                plan_summary = f"""
Test Scenarios to Cover:
{json.dumps(ctx.test_plan.test_scenarios, indent=2)}
"""
            
            refinement_feedback = ""
            if ctx.reviews:
                last_review = ctx.reviews[-1]
                refinement_feedback = f"""
Previous Review Feedback:
Missing Coverage: {', '.join(last_review.missing_coverage)}
Suggestions: {', '.join(last_review.suggestions[:3])}

Address these gaps!
"""
            
            user_prompt = f"""Scenario: {ctx.scenario}

Routes: {', '.join(routes[:10])}
Components: {', '.join(components[:10])}
{plan_summary}
{refinement_feedback}

Generate test specifications:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            
            # Extract JSON array
            specs = self._extract_json_array(content)
            
            print(f"[TestWriterAgent] Generated {len(specs)} test specifications")
            
            return specs
            
        except Exception as e:
            print(f"[TestWriterAgent] Test spec generation failed: {e}")
            return None
    
    def _extract_code_block(self, content: str) -> str:
        """Extract code from markdown code blocks"""
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
        
        return content.strip()
    
    def _extract_json_array(self, content: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response"""
        import re
        
        # Try to find JSON array in code block
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
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        
        return []


class ReviewerAgent:
    """Agent that reviews and critiques generated tests"""
    
    def __init__(self, config: MultiAgentTestConfig):
        self.config = config
        
        if not ChatLiteLLM or not HumanMessage:
            raise RuntimeError("LangChain dependencies required")
    
    def review_tests(self, ctx: GenerationContext) -> TestReview:
        """Review generated tests and provide feedback"""
        
        print(f"\n[ReviewerAgent] Reviewing generated tests...")
        
        try:
            llm = ChatLiteLLM(
                model=self.config.llm_model,
                temperature=0.3,  # Slightly higher for varied feedback
                max_tokens=1500
            )
            
            system_prompt = """You are an expert test reviewer specializing in E2E test quality.
Review the generated tests and provide constructive feedback.

Output a JSON object with this structure:
{
  "overall_score": 0.85,  // 0.0 to 1.0
  "strengths": ["Good use of data-testid selectors", "Comprehensive assertions"],
  "weaknesses": ["Missing error handling tests", "No accessibility checks"],
  "suggestions": ["Add keyboard navigation tests", "Test empty state scenarios"],
  "missing_coverage": ["Edge case: empty input", "Error: network failure"],
  "quality_issues": ["Unclear variable names", "Missing comments"]
}

Review criteria:
1. Test coverage (happy path, errors, edge cases, accessibility)
2. Code quality (readability, maintainability, best practices)
3. Assertions (comprehensive, specific, meaningful)
4. Selectors (robust, preferably data-testid)
5. Error handling
6. Comments and documentation
"""
            
            # Build review context
            test_content = ""
            if ctx.test_code:
                test_content = f"Test Code:\n{ctx.test_code[:2000]}..."
            if ctx.test_specs:
                test_content += f"\n\nTest Specs:\n{json.dumps(ctx.test_specs[:3], indent=2)}"
            
            plan_context = ""
            if ctx.test_plan:
                plan_context = f"""
Original Test Plan Coverage Goals:
{json.dumps(ctx.test_plan.coverage_goals, indent=2)}

Test Scenarios:
{json.dumps([s['name'] for s in ctx.test_plan.test_scenarios], indent=2)}
"""
            
            user_prompt = f"""Scenario: {ctx.scenario}
{plan_context}

Generated Tests:
{test_content}

Provide detailed review:"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            content = getattr(response, "content", "")
            
            # Extract JSON
            review_data = self._extract_json(content)
            
            if review_data:
                review = TestReview(
                    overall_score=float(review_data.get("overall_score", 0.7)),
                    strengths=review_data.get("strengths", []),
                    weaknesses=review_data.get("weaknesses", []),
                    suggestions=review_data.get("suggestions", []),
                    missing_coverage=review_data.get("missing_coverage", []),
                    quality_issues=review_data.get("quality_issues", [])
                )
                
                print(f"[ReviewerAgent] Review complete - Score: {review.overall_score:.2f}")
                print(f"[ReviewerAgent] Strengths: {len(review.strengths)}, Weaknesses: {len(review.weaknesses)}")
                
                return review
            
        except Exception as e:
            print(f"[ReviewerAgent] Review failed: {e}")
        
        # Fallback review
        return TestReview(
            overall_score=0.7,
            strengths=["Generated tests"],
            weaknesses=["Unable to provide detailed review"],
            suggestions=["Manual review recommended"],
            missing_coverage=[],
            quality_issues=[]
        )
    
    def _extract_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        import re
        
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        
        return None


class MultiAgentTestGenerator:
    """Orchestrator for multi-agent test generation workflow"""
    
    def __init__(self, config: Optional[MultiAgentTestConfig] = None):
        self.config = config or MultiAgentTestConfig()
        
        # Initialize agents
        self.planner = PlannerAgent(self.config) if self.config.enable_planner else None
        self.architect = TestArchitectAgent(self.config)
        self.writer = TestWriterAgent(self.config)
        self.reviewer = ReviewerAgent(self.config) if self.config.enable_reviewer else None
    
    def generate(
        self,
        structured_input: Dict[str, Any],
        subgraph: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute multi-agent test generation workflow"""
        
        print(f"\n{'='*80}")
        print("🤖 Multi-Agent Test Generation System")
        print(f"{'='*80}")
        print(f"Framework: {self.config.test_framework.value}")
        print(f"Output Format: {self.config.output_format.value}")
        print(f"Max Refinement Iterations: {self.config.max_refinement_iterations}")
        print(f"{'='*80}\n")
        
        # Initialize context
        ctx = GenerationContext(
            scenario=structured_input.get("goal", ""),
            goal=structured_input.get("goal", ""),
            structured_input=structured_input,
            subgraph=subgraph
        )
        
        # Phase 1: Planning
        if self.planner:
            ctx.test_plan = self.planner.create_test_plan(ctx)
        
        # Phase 2: Architecture
        ctx.test_architecture = self.architect.design_architecture(ctx)
        
        # Phase 3: Initial Generation
        print(f"\n{'='*40}")
        print("Phase 3: Initial Test Generation")
        print(f"{'='*40}")
        
        ctx.test_code, ctx.test_specs = self.writer.generate_tests(ctx)
        
        ctx.generation_history.append({
            "iteration": 0,
            "test_code": ctx.test_code,
            "test_specs": ctx.test_specs
        })
        
        # Phase 4: Iterative Refinement
        if self.config.enable_refinement and self.reviewer:
            print(f"\n{'='*40}")
            print("Phase 4: Iterative Refinement")
            print(f"{'='*40}")
            
            for iteration in range(1, self.config.max_refinement_iterations + 1):
                ctx.refinement_iteration = iteration
                
                print(f"\n[Refinement Iteration {iteration}]")
                
                # Review current tests
                review = self.reviewer.review_tests(ctx)
                ctx.reviews.append(review)
                
                # Check if quality threshold met
                if review.overall_score >= self.config.quality_threshold:
                    print(f"✅ Quality threshold met ({review.overall_score:.2f} >= {self.config.quality_threshold})")
                    break
                
                print(f"⚠️  Quality below threshold ({review.overall_score:.2f} < {self.config.quality_threshold})")
                print(f"   Weaknesses: {', '.join(review.weaknesses[:3])}")
                
                # Regenerate with feedback
                print(f"   Regenerating with feedback...")
                ctx.test_code, ctx.test_specs = self.writer.generate_tests(ctx)
                
                ctx.generation_history.append({
                    "iteration": iteration,
                    "test_code": ctx.test_code,
                    "test_specs": ctx.test_specs,
                    "review": {
                        "score": review.overall_score,
                        "weaknesses": review.weaknesses,
                        "suggestions": review.suggestions
                    }
                })
        
        # Build final output
        return self._build_output(ctx)
    
    def _build_output(self, ctx: GenerationContext) -> Dict[str, Any]:
        """Build final output structure"""
        
        output = {
            "scenario": ctx.scenario,
            "framework": self.config.test_framework.value,
            "output_format": self.config.output_format.value,
            "metadata": {
                "refinement_iterations": ctx.refinement_iteration,
                "final_quality_score": ctx.reviews[-1].overall_score if ctx.reviews else None,
                "nodes_analyzed": ctx.subgraph.get("summary", {}).get("node_count", 0),
                "edges_analyzed": ctx.subgraph.get("summary", {}).get("edge_count", 0)
            }
        }
        
        # Include test plan if generated
        if ctx.test_plan:
            output["test_plan"] = {
                "scenarios": ctx.test_plan.test_scenarios,
                "priority_areas": ctx.test_plan.priority_areas,
                "coverage_goals": ctx.test_plan.coverage_goals,
                "strategy": ctx.test_plan.testing_strategy,
                "complexity": ctx.test_plan.estimated_complexity
            }
        
        # Include architecture if generated
        if ctx.test_architecture:
            output["test_architecture"] = {
                "test_suites": ctx.test_architecture.test_suites,
                "shared_fixtures": ctx.test_architecture.shared_fixtures,
                "test_flow": ctx.test_architecture.test_flow,
                "assertions_strategy": ctx.test_architecture.assertions_strategy
            }
        
        # Include final tests
        if ctx.test_code:
            output["test_code"] = ctx.test_code
        if ctx.test_specs:
            output["test_specs"] = ctx.test_specs
        
        # Include reviews
        if ctx.reviews:
            output["reviews"] = [
                {
                    "iteration": i,
                    "score": review.overall_score,
                    "strengths": review.strengths,
                    "weaknesses": review.weaknesses,
                    "suggestions": review.suggestions,
                    "missing_coverage": review.missing_coverage
                }
                for i, review in enumerate(ctx.reviews)
            ]
        
        # Include generation history
        output["generation_history"] = ctx.generation_history
        
        print(f"\n{'='*80}")
        print("✅ Multi-Agent Test Generation Complete")
        print(f"{'='*80}")
        if ctx.reviews:
            print(f"Final Quality Score: {ctx.reviews[-1].overall_score:.2f}")
            print(f"Refinement Iterations: {ctx.refinement_iteration}")
        if ctx.test_plan:
            print(f"Test Scenarios: {len(ctx.test_plan.test_scenarios)}")
        if ctx.test_specs:
            print(f"Test Specifications: {len(ctx.test_specs)}")
        if ctx.test_code:
            print(f"Test Code Length: {len(ctx.test_code)} chars")
        print(f"{'='*80}\n")
        
        return output


# --------------- Public API ---------------
def multi_agent_generate(
    structured_input: Dict[str, Any],
    subgraph: Dict[str, Any],
    config: Optional[MultiAgentTestConfig] = None
) -> Dict[str, Any]:
    """Execute multi-agent test generation
    
    Args:
        structured_input: Structured scenario from Stage 1
        subgraph: KG subgraph from Stage 2
        config: Optional MultiAgentTestConfig
    
    Returns:
        Complete test generation output including tests, reviews, history
    """
    generator = MultiAgentTestGenerator(config)
    return generator.generate(structured_input, subgraph)


# --------------- CLI ---------------
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Multi-Agent Test Generation System")
    parser.add_argument("--input", type=str, required=True, help="JSON file with structured input and subgraph")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--framework", type=str, choices=["playwright", "cypress", "selenium"],
                       default="playwright", help="Test framework")
    parser.add_argument("--format", type=str, choices=["code", "spec", "both"],
                       default="both", help="Output format")
    parser.add_argument("--iterations", type=int, default=3, help="Max refinement iterations")
    parser.add_argument("--disable-planner", action="store_true", help="Disable planner agent")
    parser.add_argument("--disable-reviewer", action="store_true", help="Disable reviewer agent")
    parser.add_argument("--disable-refinement", action="store_true", help="Disable refinement")
    
    args = parser.parse_args()
    
    # Load input
    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Extract structured input and subgraph
    if "structured" in data and "nodes" in data:
        structured_input = data["structured"]
        subgraph = data
    elif "stage1" in data and "stage2a" in data:
        structured_input = data["stage1"].get("structured")
        subgraph = data["stage2a"]
    else:
        print("error: Input must contain structured input and subgraph", file=sys.stderr)
        sys.exit(1)
    
    # Build config
    config = MultiAgentTestConfig()
    config.test_framework = TestFramework(args.framework)
    config.output_format = OutputFormat(args.format)
    config.max_refinement_iterations = args.iterations
    config.enable_planner = not args.disable_planner
    config.enable_reviewer = not args.disable_reviewer
    config.enable_refinement = not args.disable_refinement
    
    # Generate tests
    result = multi_agent_generate(structured_input, subgraph, config)
    
    # Save output
    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"✅ Output saved: {args.output}", file=sys.stderr)
        
        # Save test code separately if generated
        if result.get("test_code"):
            code_file = args.output.rsplit(".", 1)[0] + "_test_code.txt"
            with open(code_file, "w") as f:
                f.write(result["test_code"])
            print(f"✅ Test code saved: {code_file}", file=sys.stderr)
    else:
        print(output_json)


