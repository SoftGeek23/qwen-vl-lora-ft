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
    ):
        """
        Initialize the reflector agent.
        
        Args:
            memory_index: The memory index to update with new insights
            reflector_model: Model to use for reflection (if None, uses rule-based)
            use_llm_reflection: Whether to use LLM for reflection or rule-based
        """
        self.memory_index = memory_index
        self.reflector_model = reflector_model
        self.use_llm_reflection = use_llm_reflection and reflector_model is not None
        
        # Statistics
        self.reflection_count = 0
        self.insights_generated = 0
    
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
    ) -> List[MemoryExemplar]:
        """
        Reflect on task logs and generate memory insights.
        
        Args:
            task_logs: List of task logs to analyze
            task_type_filter: Only reflect on logs of this task type
            
        Returns:
            List of new MemoryExemplar objects to add
        """
        # Filter by task type if specified
        if task_type_filter:
            task_logs = [log for log in task_logs if log.task_type == task_type_filter]
        
        if not task_logs:
            logger.warning("No task logs to reflect on")
            return []
        
        # Separate successes and failures
        successes = [log for log in task_logs if log.success]
        failures = [log for log in task_logs if not log.success]
        
        logger.info(f"Reflecting on {len(task_logs)} tasks: {len(successes)} successes, {len(failures)} failures")
        
        new_memories = []
        
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
        
        self.reflection_count += 1
        self.insights_generated += len(new_memories)
        
        return new_memories
    
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
                task_type=None,  # Applies to all tasks
                timestamp=time.time(),
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
            task_type=failures[0].task_type if failures else None,
            timestamp=time.time(),
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
            task_type=task_type,
            timestamp=time.time(),
        )
    
    def update_memory_index(self, new_memories: List[MemoryExemplar]):
        """Add new memories to the memory index."""
        for memory in new_memories:
            self.memory_index.add_memory(
                state_summary=memory.state_summary,
                action=memory.action,
                result=memory.result,
                reflection=memory.reflection,
                task_type=memory.task_type,
                task_id=memory.task_id,
            )
        
        logger.info(f"Added {len(new_memories)} new memory insights to index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about reflections performed."""
        return {
            "reflection_count": self.reflection_count,
            "insights_generated": self.insights_generated,
            "memory_index_size": len(self.memory_index.memories),
        }

