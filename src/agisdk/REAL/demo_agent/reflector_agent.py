"""
Reflector Agent Module

This module implements a "sleep state" reflection system where:
1. After tasks complete, logs are collected
2. A reflector agent analyzes common failures and successes
3. Insights are added to the memory index for future use
4. The agent can access these memories mid-execution

The reflector operates in a "sleep state" - after the agent completes
a batch of tasks, it reflects on the patterns and updates memory.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

from .memory_index import MemoryIndex, MemoryExemplar

logger = logging.getLogger(__name__)


@dataclass
class TaskLog:
    """Represents a single task execution log."""
    task_name: str
    task_type: Optional[str]
    task_id: Optional[str]
    success: bool
    reward: float
    elapsed_time: float
    n_steps: int
    exp_dir: str
    error: Optional[str] = None
    actions: List[str] = None
    episode_info_path: Optional[str] = None
    
    def __post_init__(self):
        if self.actions is None:
            self.actions = []


class ReflectorAgent:
    """
    Reflector agent that analyzes task logs and extracts insights.
    
    This agent:
    1. Collects logs from completed tasks
    2. Identifies common failure patterns
    3. Identifies successful strategies
    4. Generates high-level reflections
    5. Updates the memory index with new insights
    """
    
    def __init__(
        self,
        memory_index: MemoryIndex,
        reflector_model: str = None,
        use_llm_reflection: bool = True,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        """
        Initialize the reflector agent.
        
        Args:
            memory_index: The memory index to update with new insights
            reflector_model: Model to use for reflection (if None, uses rule-based)
            use_llm_reflection: Whether to use LLM for reflection or rule-based
            openai_api_key: OpenAI API key (falls back to env var)
            openrouter_api_key: OpenRouter API key (falls back to env var)
            anthropic_api_key: Anthropic API key (falls back to env var)
        """
        self.memory_index = memory_index
        self.reflector_model = reflector_model
        self.use_llm_reflection = use_llm_reflection and reflector_model is not None
        
        # Initialize LLM client if using LLM reflection
        self.llm_client = None
        if self.use_llm_reflection:
            self.llm_client = self._init_llm_client(
                reflector_model,
                openai_api_key,
                openrouter_api_key,
                anthropic_api_key,
            )
            if not self.llm_client:
                logger.warning(f"Failed to initialize LLM client for {reflector_model}, falling back to rule-based reflection")
                self.use_llm_reflection = False
        
        # Statistics
        self.reflection_count = 0
        self.insights_generated = 0
    
    def _init_llm_client(
        self,
        model_name: str,
        openai_api_key: Optional[str],
        openrouter_api_key: Optional[str],
        anthropic_api_key: Optional[str],
    ):
        """Initialize LLM client based on model name."""
        try:
            from anthropic import Anthropic
            from openai import OpenAI
            
            if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
                client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
                return ("openai", client, model_name)
            
            elif model_name.startswith("openrouter/"):
                actual_model = model_name.replace("openrouter/", "", 1)
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
                )
                return ("openrouter", client, actual_model)
            
            elif model_name.startswith("local/"):
                actual_model = model_name.replace("local/", "", 1)
                client = OpenAI(
                    base_url="http://localhost:7999/v1",
                    api_key="FEEL_THE_AGI",
                )
                return ("local", client, actual_model)
            
            elif any(model_name.startswith(prefix) for prefix in ["claude-", "sonnet-"]):
                client = Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
                return ("anthropic", client, model_name)
            
            else:
                logger.warning(f"Unknown model type: {model_name}, falling back to rule-based")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            return None
    
    def collect_task_logs(
        self,
        results: Dict[str, Any],
        results_dir: str = "./results",
    ) -> List[TaskLog]:
        """
        Collect task logs from harness results.
        
        Args:
            results: Dictionary of task results from harness.run()
            results_dir: Directory where results are stored
            
        Returns:
            List of TaskLog objects
        """
        task_logs = []
        
        for task_name, result in results.items():
            try:
                # Extract task type and ID
                task_type = None
                task_id = None
                if "." in task_name:
                    parts = task_name.split(".", 1)
                    if "-" in parts[1]:
                        task_type = parts[1].split("-")[0]
                        task_id = parts[1]
                
                # Get experiment directory (already in results/ directory)
                exp_dir = result.get("exp_dir", "")
                
                # PRIMARY: Parse experiment.log for actions and errors
                actions = []
                errors = []
                log_path = Path(exp_dir) / "experiment.log" if exp_dir else None
                
                if log_path and log_path.exists():
                    try:
                        log_text = log_path.read_text()
                        
                        # Extract actions from code blocks: ```action("args")```
                        action_pattern = r'```(\w+\([^`]+\))```'
                        action_matches = re.findall(action_pattern, log_text)
                        actions = action_matches
                        
                        # Extract errors - look for ERROR lines with error messages
                        error_pattern = r'ERROR - Error during action execution attempt: (.+?)(?=\nTraceback|\n2025-|\Z)'
                        error_matches = re.findall(error_pattern, log_text, re.DOTALL)
                        
                        # Also extract the main error type from traceback
                        for error_match in error_matches:
                            # Get the first line (main error message)
                            error_lines = error_match.strip().split('\n')
                            if error_lines:
                                main_error = error_lines[0].strip()
                                errors.append(main_error)
                        
                        # If no errors found in ERROR lines, look for exceptions in tracebacks
                        if not errors:
                            exception_pattern = r'(\w+Error|ValueError|TimeoutError|Exception):\s*(.+?)(?=\nTraceback|\nCall log|\n2025-|\Z)'
                            exception_matches = re.findall(exception_pattern, log_text, re.DOTALL)
                            for exc_type, exc_msg in exception_matches:
                                errors.append(f"{exc_type}: {exc_msg.strip()}")
                        
                        logger.debug(f"Parsed {len(actions)} actions and {len(errors)} errors from experiment.log")
                        
                    except Exception as e:
                        logger.warning(f"Failed to parse experiment.log: {e}")
                
                # FALLBACK: Try step info files if log parsing failed
                if not actions and exp_dir:
                    try:
                        from agisdk.REAL.browsergym.experiments import get_exp_result
                        exp_result = get_exp_result(exp_dir)
                        steps_info = exp_result.steps_info
                        
                        # Extract actions from step info
                        for step_info in steps_info:
                            if step_info and step_info.action:
                                actions.append(step_info.action)
                    except Exception as e:
                        logger.warning(f"Failed to load step info from {exp_dir}: {e}")
                
                # Combine errors - use first error if multiple found
                primary_error = errors[0] if errors else result.get("err_msg")
                
                task_log = TaskLog(
                    task_name=task_name,
                    task_type=task_type,
                    task_id=task_id,
                    success=result.get("cum_reward", 0) == 1,
                    reward=result.get("cum_reward", 0),
                    elapsed_time=result.get("elapsed_time", 0),
                    n_steps=result.get("n_steps", 0),
                    exp_dir=exp_dir,
                    error=primary_error,  # Use error from experiment.log parsing
                    actions=actions,
                    episode_info_path=str(log_path) if log_path and log_path.exists() else None,
                )
                task_logs.append(task_log)
            except Exception as e:
                logger.error(f"Failed to process task log for {task_name}: {e}")
        
        return task_logs
    
    def reflect_on_logs(
        self,
        task_logs: List[TaskLog],
        task_type_filter: Optional[str] = None,
    ) -> tuple[List[MemoryExemplar], List[str]]:
        """
        Reflect on task logs and generate memory insights and high-level strategies.
        
        Args:
            task_logs: List of task logs to analyze
            task_type_filter: Only reflect on logs of this task type
            
        Returns:
            Tuple of (new_memories, new_strategies):
            - new_memories: List of new MemoryExemplar objects (specific cases)
            - new_strategies: List of high-level strategy strings (for prompt injection)
        """
        # Filter by task type if specified
        if task_type_filter:
            task_logs = [log for log in task_logs if log.task_type == task_type_filter]
        
        if not task_logs:
            logger.warning("No task logs to reflect on")
            return [], []
        
        # Separate successes and failures
        successes = [log for log in task_logs if log.success]
        failures = [log for log in task_logs if not log.success]
        
        logger.info(f"Reflecting on {len(task_logs)} tasks: {len(successes)} successes, {len(failures)} failures")
        
        new_memories = []
        new_strategies = []
        
        # Use LLM-based reflection if available, otherwise fall back to rule-based
        if self.use_llm_reflection and self.llm_client:
            logger.info("Using LLM-based reflection to generate memories and strategies")
            llm_memories, llm_strategies = self._generate_llm_memories_and_strategies(task_logs, successes, failures)
            new_memories.extend(llm_memories)
            new_strategies.extend(llm_strategies)
        else:
            logger.info("Using rule-based reflection to generate memories")
            # Analyze common failure patterns
            if failures:
                failure_insights = self._analyze_failures(failures)
                new_memories.extend(failure_insights)
            
            # Analyze successful strategies
            if successes:
                success_insights = self._analyze_successes(successes)
                new_memories.extend(success_insights)
            
            # Generate cross-task insights
            cross_insights = self._generate_cross_task_insights(task_logs)
            new_memories.extend(cross_insights)
            
            # Extract high-level strategies from patterns
            strategies = self._extract_high_level_strategies(task_logs, successes, failures)
            new_strategies.extend(strategies)
        
        self.reflection_count += 1
        self.insights_generated += len(new_memories)
        
        return new_memories, new_strategies
    
    def _analyze_failures(self, failures: List[TaskLog]) -> List[MemoryExemplar]:
        """Analyze common failure patterns."""
        insights = []
        
        # Group failures by error type
        error_groups: Dict[str, List[TaskLog]] = {}
        for failure in failures:
            error_key = self._categorize_error(failure.error)
            if error_key not in error_groups:
                error_groups[error_key] = []
            error_groups[error_key].append(failure)
        
        # Generate insights for each error category
        for error_category, error_failures in error_groups.items():
            if len(error_failures) >= 2:  # Only if pattern appears multiple times
                insight = self._create_failure_insight(error_category, error_failures)
                if insight:
                    insights.append(insight)
        
        return insights
    
    def _analyze_successes(self, successes: List[TaskLog]) -> List[MemoryExemplar]:
        """Analyze successful strategies."""
        insights = []
        
        # Look for common action patterns in successful tasks
        if len(successes) >= 2:
            # Find common action sequences
            common_patterns = self._find_common_action_patterns(successes)
            
            for pattern, pattern_tasks in common_patterns.items():
                if len(pattern_tasks) >= 2:  # Pattern appears in multiple successes
                    insight = self._create_success_insight(pattern, pattern_tasks)
                    if insight:
                        insights.append(insight)
        
        return insights
    
    def _generate_cross_task_insights(self, task_logs: List[TaskLog]) -> List[MemoryExemplar]:
        """Generate insights that apply across multiple tasks."""
        insights = []
        
        # Analyze step count patterns
        avg_steps_success = sum(log.n_steps for log in task_logs if log.success) / max(len([log for log in task_logs if log.success]), 1)
        avg_steps_failure = sum(log.n_steps for log in task_logs if not log.success) / max(len([log for log in task_logs if not log.success]), 1)
        
        if avg_steps_failure > avg_steps_success * 1.5:
            # Failures tend to take longer - might indicate getting stuck
            insight = MemoryExemplar(
                state_summary="Task taking many steps without progress",
                action="Consider re-evaluating approach or checking for errors",
                result="failure",
                reflection=f"Tasks that fail often take {avg_steps_failure:.1f} steps vs {avg_steps_success:.1f} for successes. If a task is taking too long, consider changing strategy.",
            )
            insights.append(insight)
        
        return insights
    
    def _categorize_error(self, error: Optional[str]) -> str:
        """Categorize an error message."""
        if not error:
            return "unknown_error"
        
        error_lower = error.lower()
        
        if "not found" in error_lower or "detached" in error_lower:
            return "element_not_found"
        elif "not an <input>" in error_lower or "not editable" in error_lower:
            return "wrong_element_type"
        elif "multi-action" in error_lower or "multiple actions" in error_lower:
            return "multiple_actions"
        elif "timeout" in error_lower:
            return "timeout"
        elif "click" in error_lower:
            return "click_error"
        elif "fill" in error_lower:
            return "fill_error"
        else:
            return "other_error"
    
    def _create_failure_insight(
        self,
        error_category: str,
        failures: List[TaskLog],
    ) -> Optional[MemoryExemplar]:
        """Create a memory insight from failure patterns."""
        
        # Get example error
        example_error = failures[0].error or "Unknown error"
        
        # Create reflection based on error category
        reflections = {
            "element_not_found": "Element was not found or page changed. Wait for page to load or re-identify elements using updated axtree.",
            "wrong_element_type": "Tried to interact with wrong element type. Check element type before using fill() or click(). Use axtree to identify correct element.",
            "multiple_actions": "Only one action per step is allowed. Break down complex operations into individual steps.",
            "timeout": "Action timed out. Page may be loading or element may not be ready. Wait or try alternative approach.",
            "click_error": "Click action failed. Verify element is clickable and visible. Check for overlays or popups.",
            "fill_error": "Fill action failed. Ensure element is an input field and is editable. Check element type in axtree.",
        }
        
        reflection = reflections.get(
            error_category,
            f"Common error pattern: {example_error[:100]}. Review action and page state carefully."
        )
        
        # Create state summary from common context
        state_summary = f"Encountering {error_category.replace('_', ' ')} error"
        
        # Suggest action
        action_suggestions = {
            "element_not_found": "Wait for page load, then re-identify element",
            "wrong_element_type": "Check element type in axtree before action",
            "multiple_actions": "Execute one action at a time",
            "timeout": "Wait for element or try alternative approach",
            "click_error": "Verify element is clickable",
            "fill_error": "Verify element is an input field",
        }
        
        suggested_action = action_suggestions.get(
            error_category,
            "Review error and adjust approach"
        )
        
        return MemoryExemplar(
            state_summary=state_summary,
            action=suggested_action,
            result=f"failure: {error_category} (seen in {len(failures)} tasks)",
            reflection=reflection,
        )
    
    def _find_common_action_patterns(
        self,
        successes: List[TaskLog],
    ) -> Dict[str, List[TaskLog]]:
        """Find common action patterns in successful tasks."""
        patterns: Dict[str, List[TaskLog]] = {}
        
        # Look for common action sequences (simplified - can be enhanced)
        for success in successes:
            if not success.actions:
                continue
            
            # Create pattern key from first few actions
            if len(success.actions) >= 2:
                pattern_key = " -> ".join(success.actions[:2])
                if pattern_key not in patterns:
                    patterns[pattern_key] = []
                patterns[pattern_key].append(success)
        
        return patterns
    
    def _create_success_insight(
        self,
        pattern: str,
        successes: List[TaskLog],
    ) -> Optional[MemoryExemplar]:
        """Create a memory insight from successful patterns."""
        
        # Extract task type (should be same for all in pattern)
        task_type = successes[0].task_type if successes else None
        
        return MemoryExemplar(
            state_summary=f"Starting task in {task_type or 'general'} context",
            action=pattern.split(" -> ")[0] if " -> " in pattern else pattern,
            result="success",
            reflection=f"This action sequence worked well in {len(successes)} successful tasks. Consider using similar approach.",
        )
    
    def update_memory_index(self, new_memories: List[MemoryExemplar]):
        """Add new memories to the memory index."""
        for memory in new_memories:
            self.memory_index.add_memory(
                state_summary=memory.state_summary,
                action=memory.action,
                result=memory.result,
                reflection=memory.reflection,
            )
        
        logger.info(f"Added {len(new_memories)} new memory insights to index")
    
    def _generate_llm_memories_and_strategies(
        self,
        task_logs: List[TaskLog],
        successes: List[TaskLog],
        failures: List[TaskLog],
    ) -> tuple[List[MemoryExemplar], List[str]]:
        """Generate memories and strategies using LLM-based reflection on actual task logs."""
        if not self.llm_client:
            return [], []
        
        client_type, client, model_name = self.llm_client
        
        # Build detailed prompt with actual task data
        prompt = self._build_reflection_prompt(task_logs, successes, failures)
        
        try:
            if client_type == "anthropic":
                response = self._query_anthropic(client, model_name, prompt)
            else:
                response = self._query_openai(client, model_name, prompt, client_type == "openrouter")
            
            # Parse LLM response into MemoryExemplar objects and strategies
            memories, strategies = self._parse_llm_response(response, task_logs)
            logger.info(f"LLM generated {len(memories)} memory insights and {len(strategies)} strategies")
            return memories, strategies
            
        except Exception as e:
            logger.error(f"LLM reflection failed: {e}, falling back to rule-based")
            return [], []
    
    def _build_reflection_prompt(
        self,
        task_logs: List[TaskLog],
        successes: List[TaskLog],
        failures: List[TaskLog],
    ) -> str:
        """Build a detailed prompt for LLM reflection using actual task data."""
        
        prompt = """You are analyzing task execution logs to extract two types of insights for an AI agent:

1. **SPECIFIC MEMORIES**: Contextual, state-action-result exemplars for specific situations
2. **HIGH-LEVEL STRATEGIES**: General guidance and best practices for the agent's prompt

## MEMORIES (Specific Cases)
These are contextual exemplars that help the agent in specific situations. For each memory, provide:
- **state_summary**: A specific, contextual description of the page state or situation (include URL, key elements, goal context)
- **action**: The specific action taken (or that should be taken)
- **result**: "success" or "failure: [specific error message]"
- **reflection**: A detailed explanation of why it worked/didn't work, what was learned, and how to apply this knowledge

Focus on:
- Specific, actionable insights (not generic advice)
- Real error messages and page states from the logs
- Patterns that appear multiple times
- Concrete guidance the agent can follow

## STRATEGIES (High-Level Guidance)
CRITICAL: Strategies must be UNIVERSAL and NON-CONTEXTUAL. They apply to ALL tasks, regardless of context.

CRITERIA FOR STRATEGIES:
- Must be applicable to ANY task type (not specific to e-commerce, forms, dropdowns, etc.)
- Must NOT reference specific page states, element types, task contexts, or scenarios
- Must be fundamental behavioral rules (not "when X happens, do Y")
- Should be 5-10 core principles maximum
- Must NOT contain contextual phrases like "when working with", "for [specific]", "on [site]", "when encountering [specific]"

VALID STRATEGY EXAMPLES:
✓ "Execute only one action per turn"
✓ "Always wait for environment response before proceeding"
✓ "Use send_msg_to_user() to communicate results to the user"
✓ "After an error, try a different approach rather than repeating"

INVALID (These should be MEMORIES, not strategies):
✗ "When working with dropdowns, use click()..." (contextual - mentions specific element type)
✗ "For e-commerce sites, follow pattern X..." (contextual - mentions specific site type)
✗ "When on page Y, do Z..." (contextual - mentions specific page state)
✗ "For product search tasks..." (contextual - mentions specific task type)
✗ "When encountering element-not-found errors..." (contextual - mentions specific error type)

Generate AT MOST 2-3 new strategies per reflection session. Only add if it's truly universal and non-contextual. If unsure, make it a MEMORY instead.

Return your analysis as a JSON object with two arrays:
{
  "memories": [
    {
      "state_summary": "...",
      "action": "...",
      "result": "...",
      "reflection": "..."
    }
  ],
  "strategies": [
    "Strategy 1: ...",
    "Strategy 2: ..."
  ]
}

"""
        
        # Add failure examples with actual data
        if failures:
            prompt += "\n## FAILURE PATTERNS TO ANALYZE\n\n"
            for i, failure in enumerate(failures[:10], 1):  # Limit to 10 for prompt size
                prompt += f"### Failure {i}: {failure.task_name}\n"
                prompt += f"- Task Type: {failure.task_type or 'unknown'}\n"
                prompt += f"- Steps: {failure.n_steps}\n"
                if failure.error:
                    prompt += f"- Error: {failure.error[:300]}\n"
                if failure.actions:
                    prompt += f"- Actions taken: {', '.join(failure.actions[:5])}\n"
                prompt += "\n"
        
        # Add success examples with actual data
        if successes:
            prompt += "\n## SUCCESS PATTERNS TO ANALYZE\n\n"
            for i, success in enumerate(successes[:10], 1):  # Limit to 10 for prompt size
                prompt += f"### Success {i}: {success.task_name}\n"
                prompt += f"- Task Type: {success.task_type or 'unknown'}\n"
                prompt += f"- Steps: {success.n_steps}\n"
                if success.actions:
                    prompt += f"- Actions taken: {', '.join(success.actions[:5])}\n"
                prompt += "\n"
        
        prompt += """
Analyze these patterns and generate:
- 3-8 high-quality MEMORY insights (contextual, specific situations)
- 0-3 STRATEGY insights (only if truly universal and non-contextual)

For MEMORIES, focus on:
1. Common failure modes with specific error messages
2. Successful action sequences that worked
3. Specific page states or contexts where mistakes occurred
4. Actionable guidance based on real examples

For STRATEGIES, only include if:
- It applies to ALL tasks universally
- It contains NO contextual references
- It's a fundamental behavioral rule

Return ONLY a valid JSON object with "memories" and "strategies" arrays, no other text.
"""
        
        return prompt
    
    def _query_openai(self, client, model_name: str, prompt: str, is_openrouter: bool = False) -> str:
        """Query OpenAI-compatible API."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing AI agent behavior and extracting actionable insights. Always return valid JSON arrays."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        if is_openrouter:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent reflection
            )
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.3,
            )
        
        return response.choices[0].message.content
    
    def _query_anthropic(self, client, model_name: str, prompt: str) -> str:
        """Query Anthropic API with support for thinking mode."""
        # Handle thinking mode (e.g., "sonnet-3.7:thinking")
        ANTHROPIC_MODELS = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-opus-4": "claude-opus-4-20250514",
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "sonnet-3.7": "claude-3-7-sonnet-20250219",
        }
        
        # Parse model name and thinking mode
        base_model_name = model_name.replace(":thinking", "")
        thinking_enabled = model_name.endswith(":thinking")
        
        # Get the actual model ID
        if base_model_name in ANTHROPIC_MODELS:
            actual_model_id = ANTHROPIC_MODELS[base_model_name]
        else:
            # If not in mapping, assume it's a direct model ID
            actual_model_id = base_model_name
        
        # Configure thinking based on model capabilities and user request
        if thinking_enabled:
            thinking_budget = 10000
            thinking = {"type": "enabled", "budget_tokens": thinking_budget}
            # max_tokens must be greater than thinking.budget_tokens
            max_tokens = thinking_budget + 2000  # 12000 total
        else:
            thinking = {"type": "disabled"}
            max_tokens = 2000
        
        # When thinking is enabled, temperature must be 1.0
        temperature = 1.0 if thinking_enabled else 0.3
        
        response = client.messages.create(
            model=actual_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            system="You are an expert at analyzing AI agent behavior and extracting actionable insights. Always return valid JSON arrays.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract text content (skip thinking blocks)
        for content_block in response.content:
            if content_block.type == "text":
                return content_block.text
            elif content_block.type == "thinking":
                # Log thinking but don't return it
                logger.debug(f"Thinking block: {content_block.thinking[:100] if hasattr(content_block, 'thinking') else '...'}...")
        
        raise ValueError("No text content in Anthropic response")
    
    def _parse_llm_response(self, response: str, task_logs: List[TaskLog]) -> tuple[List[MemoryExemplar], List[str]]:
        """Parse LLM response into MemoryExemplar objects and strategies."""
        memories = []
        strategies = []
        
        try:
            # Try to extract JSON from response (might have markdown code blocks)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Handle both old format (array) and new format (object with memories and strategies)
            if isinstance(data, list):
                # Old format: just memories array
                for item in data:
                    if isinstance(item, dict) and all(key in item for key in ["state_summary", "action", "result", "reflection"]):
                        memory = MemoryExemplar(
                            state_summary=item["state_summary"],
                            action=item["action"],
                            result=item["result"],
                            reflection=item["reflection"],
                        )
                        memories.append(memory)
            elif isinstance(data, dict):
                # New format: object with memories and strategies
                if "memories" in data and isinstance(data["memories"], list):
                    for item in data["memories"]:
                        if isinstance(item, dict) and all(key in item for key in ["state_summary", "action", "result", "reflection"]):
                            memory = MemoryExemplar(
                                state_summary=item["state_summary"],
                                action=item["action"],
                                result=item["result"],
                                reflection=item["reflection"],
                            )
                            memories.append(memory)
                
                if "strategies" in data and isinstance(data["strategies"], list):
                    strategies = [s for s in data["strategies"] if isinstance(s, str) and s.strip()]
            else:
                logger.warning("LLM response is not a valid JSON array or object")
                return [], []
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        return memories, strategies
    
    def _extract_high_level_strategies(
        self,
        task_logs: List[TaskLog],
        successes: List[TaskLog],
        failures: List[TaskLog],
    ) -> List[str]:
        """Extract high-level strategies from task patterns (rule-based)."""
        strategies = []
        
        # Analyze task completion patterns
        if successes:
            # Check if successful tasks used send_msg_to_user or similar
            strategies.append("When a task asks to 'display', 'show', or 'find' information, use send_msg_to_user() to communicate the result to complete the task.")
        
        # Analyze failure patterns for high-level guidance
        incomplete_tasks = [log for log in failures if log.n_steps >= 15]
        if incomplete_tasks:
            strategies.append("If you've taken many steps without progress, re-evaluate your approach. Consider using send_msg_to_user() if you've found the information but the task isn't completing.")
        
        # Analyze action loop patterns
        loop_tasks = [log for log in failures if log.actions and len(set(log.actions[-3:])) == 1]
        if loop_tasks:
            strategies.append("When an action doesn't produce the expected result after 2-3 attempts, try a different approach instead of repeating the same action.")
        
        return strategies
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about reflections performed."""
        return {
            "reflection_count": self.reflection_count,
            "insights_generated": self.insights_generated,
            "memory_index_size": len(self.memory_index.memories),
        }

