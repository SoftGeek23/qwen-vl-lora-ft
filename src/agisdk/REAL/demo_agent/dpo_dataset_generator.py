"""
DPO Dataset Generator Module

This module generates Direct Preference Optimization (DPO) datasets from experiment logs.
It analyzes completed task runs, extracts state-action pairs, and creates preference pairs
based on success/failure outcomes for fine-tuning language models.

DPO datasets are used to train models to prefer successful actions over failed ones.
"""

import json
import logging
import gzip
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re

logger = logging.getLogger(__name__)


@dataclass
class DPOExample:
    """A single DPO training example."""
    prompt: str  # The state/context
    chosen: str  # The preferred action/response
    rejected: str  # The non-preferred action/response
    metadata: Optional[Dict[str, Any]] = None  # Additional info (task_name, step, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Only include prompt, chosen, and rejected - no metadata
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


class DPODatasetGenerator:
    """
    Generates DPO datasets from experiment logs and step info.
    
    This class:
    1. Scans results directories for completed experiments
    2. Extracts state-action pairs from step info
    3. Creates preference pairs (successful vs failed actions)
    4. Outputs DPO-formatted JSONL files
    
    Can work in two modes:
    - Direct mode: Generates pairs from step data directly
    - Reflection-based mode: Uses failure reflections from ReflectorAgent + Claude thinking
    """
    
    def __init__(
        self,
        results_dir: str = "./results",
        output_file: str = "./dpo_dataset.jsonl",
        min_steps: int = 1,
        include_failed_only: bool = False,
        dpo_model: str = "sonnet-3.7:thinking",
        anthropic_api_key: Optional[str] = None,
    ):
        """
        Initialize the DPO dataset generator.
        
        Args:
            results_dir: Directory containing experiment results
            output_file: Path to output JSONL file
            min_steps: Minimum number of steps required for a task to be included
            include_failed_only: If True, only include examples from failed tasks
            dpo_model: Model to use for DPO generation (default: Claude Sonnet 3.7 thinking)
            anthropic_api_key: Anthropic API key (falls back to env var)
        """
        self.results_dir = Path(results_dir)
        self.output_file = Path(output_file)
        self.min_steps = min_steps
        self.include_failed_only = include_failed_only
        self.dpo_model = dpo_model
        self.anthropic_api_key = anthropic_api_key
        
        # Initialize Claude client for DPO generation
        self.claude_client = None
        if dpo_model and ("claude" in dpo_model.lower() or "sonnet" in dpo_model.lower()):
            self._init_claude_client()
        
        # Cache for task configs and action sets
        self._task_config_cache = {}
        self._action_set_description = None
        
    def _find_experiment_dirs(self) -> List[Path]:
        """Find all experiment directories in results_dir."""
        exp_dirs = []
        
        if not self.results_dir.exists():
            logger.warning(f"Results directory {self.results_dir} does not exist")
            return exp_dirs
        
        # Walk through all directories
        for root, dirs, files in os.walk(self.results_dir):
            root_path = Path(root)
            
            # Check if this directory has summary_info.json (indicates an experiment)
            if "summary_info.json" in files:
                exp_dirs.append(root_path)
        
        logger.info(f"Found {len(exp_dirs)} experiment directories")
        return exp_dirs
    
    def _init_claude_client(self):
        """Initialize Claude client for DPO generation."""
        try:
            import os
            from anthropic import Anthropic
            
            api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY is required for DPO generation with Claude models")
                logger.error("Please set the environment variable: export ANTHROPIC_API_KEY='your-key'")
                return
            
            # Map model names
            model_mapping = {
                "sonnet-3.7": "claude-3-7-sonnet-20250219",
                "sonnet-3.7:thinking": "claude-3-7-sonnet-20250219",
                "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
            }
            
            base_model = self.dpo_model.replace(":thinking", "")
            actual_model = model_mapping.get(base_model, base_model)
            
            self.claude_client = Anthropic(api_key=api_key)
            self.claude_model = actual_model
            self.use_thinking = ":thinking" in self.dpo_model.lower()
            
            logger.info(f"Initialized Claude client for DPO generation with model {actual_model}")
        except ImportError as e:
            logger.error(f"Failed to import anthropic library: {e}")
            logger.error("Please install it with: pip install anthropic")
            self.claude_client = None
            self.claude_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize Claude client: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.claude_client = None
            self.claude_model = None
    
    def _load_summary_info(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """Load summary_info.json from experiment directory."""
        summary_path = exp_dir / "summary_info.json"
        if not summary_path.exists():
            return None
        
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load summary_info from {exp_dir}: {e}")
            return None
    
    def _load_task_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Load task.json configuration file to get goal, evaluation rubric, etc.
        
        Args:
            task_name: Task name like "v1.omnizon-1" or "omnizon-1"
            
        Returns:
            Task config dict with goal, evals, etc., or None if not found
        """
        # Check cache first
        if task_name in self._task_config_cache:
            return self._task_config_cache[task_name]
        
        try:
            # Parse version and task id
            if "." in task_name:
                version, task_id = task_name.split(".", 1)
            else:
                version = "v1"
                task_id = task_name
            
            # Try to load using TaskConfig
            from agisdk.REAL.browsergym.webclones.task_config import TaskConfig
            task_config = TaskConfig(task_id, version=version)
            
            config = {
                "goal": task_config.get_goal(),
                "evals": [eval_config.to_json() for eval_config in task_config.get_evals()],
                "id": task_config.get_task_id(),
                "version": version,
            }
            
            self._task_config_cache[task_name] = config
            return config
        except Exception as e:
            logger.debug(f"Could not load task config for {task_name}: {e}")
            return None
    
    def _get_action_set_description(self) -> str:
        """
        Get description of all available actions from HighLevelActionSet.
        
        Returns:
            Formatted string describing all available actions
        """
        if self._action_set_description:
            return self._action_set_description
        
        try:
            from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
            
            # Create action set with default subsets (same as agent uses)
            action_set = HighLevelActionSet(
                subsets=["chat", "infeas", "bid", "nav", "tab"],
                multiaction=False,  # Agent uses single actions
            )
            
            # Get description with examples
            description = action_set.describe(with_long_description=True, with_examples=True)
            
            self._action_set_description = description
            return description
        except Exception as e:
            logger.warning(f"Could not load action set description: {e}")
            # Fallback to basic list
            return """Available actions:
- click("bid") - Click an element by its bid
- fill("bid", "text") - Fill a text input field
- scroll("up"|"down"|"left"|"right") - Scroll the page
- send_msg_to_user("message") - Send a message to the user (use when task asks to display/show/find/report)
- select_option("bid", "value") - Select an option from a dropdown
- focus("bid") - Focus on an element
- goto("url") - Navigate to a URL
- noop() or noop(milliseconds) - Wait/no operation
- report_infeasible("reason") - Report task as infeasible"""
    
    def _load_step_info(self, exp_dir: Path, step: int) -> Optional[Any]:
        """Load step info from pickle file."""
        step_file = exp_dir / f"step_{step}.pkl.gz"
        if not step_file.exists():
            return None
        
        try:
            with gzip.open(step_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load step {step} from {exp_dir}: {e}")
            return None
    
    def _extract_state_summary(self, step_info) -> str:
        """
        Extract a textual summary of the state from step_info observation.
        
        This creates the "prompt" for DPO training - the context the agent sees.
        """
        if not step_info or not hasattr(step_info, 'obs') or not step_info.obs:
            return ""
        
        obs = step_info.obs
        state_parts = []
        
        # Goal/instruction - CRITICAL, put first and emphasize
        goal_text = None
        if obs.get("goal_object"):
            goal = obs["goal_object"]
            if isinstance(goal, list) and len(goal) > 0:
                if isinstance(goal[0], dict) and "text" in goal[0]:
                    goal_text = goal[0]['text']
            elif isinstance(goal, str):
                goal_text = goal
        
        if goal_text:
            state_parts.append(f"## TASK GOAL (CRITICAL - READ CAREFULLY):")
            state_parts.append(f"{goal_text}")
            state_parts.append("")
            # Highlight key action verbs
            goal_lower = goal_text.lower()
            if any(keyword in goal_lower for keyword in ["display", "show", "find", "report", "tell"]):
                state_parts.append("⚠️ IMPORTANT: This task requires COMMUNICATING information to the user using send_msg_to_user()")
            state_parts.append("")
        
        # URL
        if obs.get("url"):
            state_parts.append(f"Current URL: {obs['url']}")
        
        # Accessibility tree (first 500 chars)
        if obs.get("axtree_txt"):
            axtree_snippet = obs["axtree_txt"][:500]
            state_parts.append(f"Page elements: {axtree_snippet}")
        
        # HTML (first 300 chars if available)
        if obs.get("pruned_html"):
            html_snippet = obs["pruned_html"][:300]
            state_parts.append(f"HTML: {html_snippet}")
        
        # Last action error (if any) - important context
        if obs.get("last_action_error"):
            state_parts.append(f"Previous error: {obs['last_action_error'][:200]}")
        
        # Action history (last 2 actions for context)
        if obs.get("action_history") and len(obs["action_history"]) > 0:
            recent_actions = obs["action_history"][-2:]
            state_parts.append(f"Recent actions: {', '.join(recent_actions)}")
        
        return "\n".join(state_parts)
    
    def _extract_action(self, step_info) -> Optional[str]:
        """Extract the actual executable action taken at this step."""
        if not step_info:
            return None
        
        # Action is stored in step_info.action - this is the real executed action
        if hasattr(step_info, 'action') and step_info.action:
            action = step_info.action
            # Ensure it's a real executable action (not abstract reasoning)
            if isinstance(action, str) and self._is_valid_executable_action(action):
                return action
        
        return None
    
    def _extract_trajectory(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Extract full trajectory (sequence of actions) from an experiment.
        
        Returns:
            Dict with:
            - actions: List of actual executed actions in order
            - states: List of state summaries at each step
            - success: Whether task succeeded
            - n_steps: Number of steps
            - invalid_actions: Count of invalid/failed actions
            - exp_dir: Experiment directory
            - task_name: Task name
        """
        summary_info = self._load_summary_info(exp_dir)
        if not summary_info:
            logger.debug(f"No summary_info found in {exp_dir}")
            return None
        
        steps = self._load_all_steps(exp_dir)
        actions = []
        states = []
        invalid_actions = 0
        
        if steps:
            # Extract from step_info files
            for step_info in steps:
                # Extract actual executable action
                action = self._extract_action(step_info)
                if action:
                    actions.append(action)
                else:
                    # Count as invalid if no valid action
                    invalid_actions += 1
                
                # Extract state
                state = self._extract_state_summary(step_info)
                if state:
                    states.append(state)
                
                # Check for action errors
                if step_info.obs and step_info.obs.get("last_action_error"):
                    invalid_actions += 1
        else:
            # Fallback: parse from experiment.log
            logger.debug(f"No step files found in {exp_dir}, trying experiment.log")
            log_actions, errors = self._parse_experiment_log(exp_dir)
            if log_actions:
                actions = log_actions
                invalid_actions = len(errors)
                
                # Try to extract state information from experiment files
                task_name = summary_info.get("task_name", "unknown")
                base_state = f"Task: {task_name}"
                
                # Try to load goal_object if available
                goal_text = None
                goal_object_path = exp_dir / "goal_object.pkl.gz"
                if goal_object_path.exists():
                    try:
                        import gzip
                        import pickle
                        with gzip.open(goal_object_path, 'rb') as f:
                            goal_object = pickle.load(f)
                            # Extract goal text
                            if isinstance(goal_object, list) and len(goal_object) > 0:
                                if isinstance(goal_object[0], dict) and "text" in goal_object[0]:
                                    goal_text = goal_object[0]["text"]
                            elif isinstance(goal_object, str):
                                goal_text = goal_object
                    except Exception as e:
                        logger.debug(f"Could not load goal_object: {e}")
                
                # Also try to load from task.json if we have task name
                if not goal_text and task_name != "unknown":
                    task_config = self._load_task_config(task_name)
                    if task_config and task_config.get("goal"):
                        goal_text = task_config["goal"]
                
                # Also try to extract goal from experiment.log
                if not goal_text:
                    log_path = exp_dir / "experiment.log"
                    if log_path.exists():
                        try:
                            log_text = log_path.read_text()
                            # Look for goal descriptions in the log
                            goal_patterns = [
                                r'goal[:\s]+(.+?)(?=\n|The|I|You)',
                                r'search for[:\s]+"([^"]+)"',
                                r'find[:\s]+(.+?)(?=\n|and|on)',
                            ]
                            for pattern in goal_patterns:
                                matches = re.findall(pattern, log_text, re.IGNORECASE | re.DOTALL)
                                if matches:
                                    goal_text = matches[0].strip()[:300]
                                    if goal_text and len(goal_text) > 10:
                                        break
                        except Exception as e:
                            logger.debug(f"Could not extract goal from log: {e}")
                
                # Add goal prominently to base state
                if goal_text:
                    base_state = f"## TASK GOAL (CRITICAL - READ CAREFULLY):\n{goal_text[:500]}\n\n" + base_state
                    # Highlight key action verbs
                    goal_lower = goal_text.lower()
                    if any(keyword in goal_lower for keyword in ["display", "show", "find", "report", "tell"]):
                        base_state += "\n⚠️ IMPORTANT: This task requires COMMUNICATING information to the user using send_msg_to_user()\n"
                
                # Create state for each action
                states = [base_state] * len(actions)
            else:
                logger.debug(f"Could not extract actions from {exp_dir}")
                return None
        
        if not actions:
            logger.debug(f"No actions found in trajectory from {exp_dir}")
            return None
        
        return {
            "actions": actions,
            "states": states,
            "success": summary_info.get("cum_reward", 0) == 1,
            "n_steps": len(steps) if steps else len(actions),
            "invalid_actions": invalid_actions,
            "exp_dir": str(exp_dir),
            "task_name": summary_info.get("task_name", "unknown"),
        }
    
    def _load_trajectories_from_experiments(self) -> List[Dict[str, Any]]:
        """Load all trajectories from experiment directories."""
        trajectories = []
        exp_dirs = self._find_experiment_dirs()
        
        for exp_dir in exp_dirs:
            trajectory = self._extract_trajectory(exp_dir)
            if trajectory and trajectory["actions"]:  # Only include if has actions
                trajectories.append(trajectory)
        
        logger.info(f"Loaded {len(trajectories)} trajectories")
        return trajectories
    
    def _rank_trajectories(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank trajectories by outcome quality.
        
        Ranking criteria (in order):
        1. Task success (successful > failed)
        2. Fewer invalid actions (better)
        3. Fewer steps (more efficient)
        
        Returns trajectories sorted from best to worst.
        """
        def trajectory_score(traj):
            # Higher score = better trajectory
            score = 0
            # Success is most important
            if traj["success"]:
                score += 10000
            # Fewer invalid actions is better
            score += 1000 - (traj["invalid_actions"] * 100)
            # Fewer steps is better (but less important)
            score += 100 - traj["n_steps"]
            return score
        
        ranked = sorted(trajectories, key=trajectory_score, reverse=True)
        return ranked
    
    def _create_trajectory_pairs(self, ranked_trajectories: List[Dict[str, Any]]) -> List[DPOExample]:
        """
        Create DPO pairs by comparing better vs worse trajectories at similar states.
        
        For each worse trajectory, find a better one with similar initial state,
        then compare the actions taken at that state.
        """
        examples = []
        
        # Separate successful and failed trajectories
        successful = [t for t in ranked_trajectories if t["success"]]
        failed = [t for t in ranked_trajectories if not t["success"]]
        
        if not successful or not failed:
            logger.warning("Need both successful and failed trajectories to create pairs")
            return examples
        
        # For each failed trajectory, try to find a similar successful one
        for failed_traj in failed:
            if not failed_traj["states"] or not failed_traj["actions"]:
                continue
            
            # Get initial state and first action from failed trajectory
            failed_state = failed_traj["states"][0] if failed_traj["states"] else ""
            failed_action = failed_traj["actions"][0] if failed_traj["actions"] else ""
            
            if not failed_state or not failed_action:
                continue
            
            # Find best matching successful trajectory
            best_match = None
            best_similarity = 0
            
            for success_traj in successful:
                if not success_traj["states"] or not success_traj["actions"]:
                    continue
                
                success_state = success_traj["states"][0]
                success_action = success_traj["actions"][0]
                
                if not success_state or not success_action:
                    continue
                
                # Calculate state similarity
                similarity = self._calculate_state_similarity(failed_state, success_state)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (success_state, success_action)
            
            # Create DPO example if we found a match
            if best_match and best_similarity > 0.2:  # Minimum similarity threshold
                success_state, success_action = best_match
                example = DPOExample(
                    prompt=failed_state,  # Use failed state as prompt
                    chosen=success_action,  # Better action from successful trajectory
                    rejected=failed_action,  # Worse action from failed trajectory
                    metadata=None  # No metadata in final output
                )
                examples.append(example)
        
        return examples
    
    def _load_all_steps(self, exp_dir: Path) -> List[Any]:
        """Load all step info files from an experiment directory."""
        steps = []
        step_num = 0
        
        while True:
            step_info = self._load_step_info(exp_dir, step_num)
            if step_info is None:
                break
            steps.append(step_info)
            step_num += 1
        
        return steps
    
    def _parse_experiment_log(self, exp_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Parse experiment.log to extract actions and errors.
        
        Returns:
            Tuple of (actions, errors) lists
        """
        log_path = exp_dir / "experiment.log"
        if not log_path.exists():
            return [], []
        
        try:
            log_text = log_path.read_text()
            
            # Extract actions from code blocks: ```action("args")``` or ```action()```
            # Handle both single and double quotes, and actions with no args
            action_pattern = r'```(\w+\([^`]*\))```'
            action_matches = re.findall(action_pattern, log_text)
            
            # Also try without code blocks (in case format is different)
            if not action_matches:
                action_pattern2 = r'(\w+\([^\)]*\))'
                action_matches2 = re.findall(action_pattern2, log_text)
                # Filter to only valid executable actions
                action_matches = [a for a in action_matches2 if self._is_valid_executable_action(a)]
            
            # Extract errors
            errors = []
            error_pattern = r'ERROR - Error during action execution attempt: (.+?)(?=\nTraceback|\n2025-|\Z)'
            error_matches = re.findall(error_pattern, log_text, re.DOTALL)
            
            for error_match in error_matches:
                error_lines = error_match.strip().split('\n')
                if error_lines:
                    errors.append(error_lines[0].strip())
            
            if not errors:
                exception_pattern = r'(\w+Error|ValueError|TimeoutError|Exception):\s*(.+?)(?=\nTraceback|\nCall log|\n2025-|\Z)'
                exception_matches = re.findall(exception_pattern, log_text, re.DOTALL)
                for exc_type, exc_msg in exception_matches:
                    errors.append(f"{exc_type}: {exc_msg.strip()}")
            
            return action_matches, errors
            
        except Exception as e:
            logger.warning(f"Failed to parse experiment.log from {exp_dir}: {e}")
            return [], []
    
    def _extract_full_responses_from_log(self, exp_dir: Path) -> List[str]:
        """
        Extract full agent responses (reasoning + action) from experiment.log.
        
        Returns:
            List of full response strings in format: "reasoning text\n\n```action(\"args\")```"
        """
        log_path = exp_dir / "experiment.log"
        if not log_path.exists():
            return []
        
        try:
            log_text = log_path.read_text()
            responses = []
            
            # Pattern to match: "action:" line, then reasoning text, then ```action("args")```
            # The log format is:
            # 2025-12-21 ... - INFO - action:
            # [reasoning text]
            # ```action("args")```
            
            # Split by "action:" markers
            action_sections = re.split(r'\d{4}-\d{2}-\d{2}.*?INFO.*?action:\s*\n', log_text)
            
            for section in action_sections[1:]:  # Skip first section (before first action)
                # Find the code block with action
                action_match = re.search(r'```(\w+\([^`]*\))```', section)
                if action_match:
                    # Extract reasoning text before the code block
                    reasoning = section[:action_match.start()].strip()
                    action = action_match.group(1)
                    
                    # Combine into full response
                    if reasoning:
                        full_response = f"{reasoning}\n\n```{action}```"
                    else:
                        full_response = f"```{action}```"
                    
                    responses.append(full_response)
            
            return responses
            
        except Exception as e:
            logger.warning(f"Failed to extract full responses from {exp_dir}: {e}")
            return []
    
    def _create_preference_pairs(
        self,
        successful_steps: List[Tuple[Any, Dict[str, Any]]],
        failed_steps: List[Tuple[Any, Dict[str, Any]]],
    ) -> List[DPOExample]:
        """
        Create DPO preference pairs from successful and failed steps.
        
        Uses multiple strategies:
        1. Same-task pairing: Match steps from same task name (best quality)
        2. Similar-state pairing: Match steps with similar state contexts
        3. Task-type pairing: Match steps from same task type
        
        Args:
            successful_steps: List of (step_info, metadata) tuples from successful tasks
            failed_steps: List of (step_info, metadata) tuples from failed tasks
            
        Returns:
            List of DPOExample objects
        """
        examples = []
        used_successful_indices = set()
        
        # Strategy 1: Same-task pairing (highest quality)
        # Group by task name and pair steps from same task
        task_groups = {}
        for idx, (step_info, metadata) in enumerate(successful_steps + failed_steps):
            task_name = metadata.get("task_name", "unknown")
            if task_name not in task_groups:
                task_groups[task_name] = {"success": [], "failed": []}
            
            is_success = metadata.get("success", False)
            if is_success:
                task_groups[task_name]["success"].append((idx, step_info, metadata))
            else:
                task_groups[task_name]["failed"].append((idx, step_info, metadata))
        
        # Create pairs from same tasks
        for task_name, groups in task_groups.items():
            if groups["success"] and groups["failed"]:
                # Pair each failed step with a successful step from same task
                for failed_idx, failed_step_info, failed_metadata in groups["failed"]:
                    failed_state = self._extract_state_summary(failed_step_info)
                    failed_action = self._extract_action(failed_step_info)
                    
                    if not failed_state or not failed_action:
                        continue
                    
                    # Find best matching successful step from same task
                    best_match = None
                    best_match_idx = None
                    best_similarity = 0
                    
                    for success_idx, success_step_info, success_metadata in groups["success"]:
                        if success_idx in used_successful_indices:
                            continue
                        
                        success_state = self._extract_state_summary(success_step_info)
                        success_action = self._extract_action(success_step_info)
                        
                        if not success_state or not success_action:
                            continue
                        
                        # Calculate similarity
                        similarity = self._calculate_state_similarity(failed_state, success_state)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = (success_step_info, success_metadata)
                            best_match_idx = success_idx
                    
                    # Use best match or first available
                    if not best_match and groups["success"]:
                        success_idx, success_step_info, success_metadata = groups["success"][0]
                        if success_idx not in used_successful_indices:
                            best_match = (success_step_info, success_metadata)
                            best_match_idx = success_idx
                    
                    if best_match:
                        chosen_step_info, chosen_metadata = best_match
                        chosen_state = self._extract_state_summary(chosen_step_info)
                        chosen_action = self._extract_action(chosen_step_info)
                        
                        if chosen_state and chosen_action:
                            used_successful_indices.add(best_match_idx)
                            # Only create example if both actions are real executable actions
                            if chosen_action and failed_action:
                                example = DPOExample(
                                    prompt=failed_state,  # Use failed state as prompt (what agent saw)
                                    chosen=chosen_action,  # Real executable action from successful trajectory
                                    rejected=failed_action,  # Real executable action from failed trajectory
                                    metadata=None  # No metadata in final output
                                )
                                examples.append(example)
        
        # Strategy 2: Cross-task pairing for remaining failed steps
        # Find failed steps that weren't paired in Strategy 1
        paired_failed_tasks = set()
        for task_name, groups in task_groups.items():
            if groups["success"] and groups["failed"]:
                paired_failed_tasks.add(task_name)
        
        remaining_failed = [
            (step_info, metadata)
            for step_info, metadata in failed_steps
            if metadata.get("task_name", "unknown") not in paired_failed_tasks
        ]
        
        for failed_step_info, failed_metadata in remaining_failed:
            failed_state = self._extract_state_summary(failed_step_info)
            failed_action = self._extract_action(failed_step_info)
            
            if not failed_state or not failed_action:
                continue
            
            # Find best matching successful step
            best_match = None
            best_similarity = 0
            
            for success_idx, (success_step_info, success_metadata) in enumerate(successful_steps):
                if success_idx in used_successful_indices:
                    continue
                
                success_state = self._extract_state_summary(success_step_info)
                success_action = self._extract_action(success_step_info)
                
                if not success_state or not success_action:
                    continue
                
                similarity = self._calculate_state_similarity(failed_state, success_state)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (success_step_info, success_metadata, success_idx)
            
            if best_match:
                chosen_step_info, chosen_metadata, chosen_idx = best_match
                chosen_state = self._extract_state_summary(chosen_step_info)
                chosen_action = self._extract_action(chosen_step_info)
                
                if chosen_state and chosen_action:
                    used_successful_indices.add(chosen_idx)
                    # Only create example if both actions are real executable actions
                    if chosen_action and failed_action:
                        example = DPOExample(
                            prompt=failed_state,
                            chosen=chosen_action,  # Real executable action from successful trajectory
                            rejected=failed_action,  # Real executable action from failed trajectory
                            metadata=None  # No metadata in final output
                        )
                        examples.append(example)
        
        return examples
    
    def _calculate_state_similarity(self, state1: str, state2: str) -> float:
        """
        Calculate similarity score between two states (0.0 to 1.0).
        
        Uses Jaccard similarity on words, with bonus for matching key terms.
        """
        if not state1 or not state2:
            return 0.0
        
        words1 = set(state1.lower().split())
        words2 = set(state2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0
        
        # Bonus for matching key terms (URL, goal keywords)
        key_terms1 = {w for w in words1 if len(w) > 4}  # Longer words are more meaningful
        key_terms2 = {w for w in words2 if len(w) > 4}
        key_overlap = len(key_terms1 & key_terms2) / max(len(key_terms1 | key_terms2), 1)
        
        # Weighted combination
        similarity = 0.7 * jaccard + 0.3 * key_overlap
        
        return similarity
    
    
    def generate_dataset(self) -> List[DPOExample]:
        """
        Generate DPO dataset from all experiments in results_dir.
        
        Returns:
            List of DPOExample objects
        """
        exp_dirs = self._find_experiment_dirs()
        
        successful_steps = []
        failed_steps = []
        
        for exp_dir in exp_dirs:
            summary_info = self._load_summary_info(exp_dir)
            if not summary_info:
                continue
            
            # Check if task was successful
            success = summary_info.get("cum_reward", 0) == 1
            task_name = summary_info.get("task_name", "unknown")
            n_steps = summary_info.get("n_steps", 0)
            
            if n_steps < self.min_steps:
                continue
            
            # Skip successful tasks if only including failed ones
            if self.include_failed_only and success:
                continue
            
            # Load all steps
            steps = self._load_all_steps(exp_dir)
            
            if not steps:
                # Fallback: try parsing experiment.log
                actions, errors = self._parse_experiment_log(exp_dir)
                if actions:
                    # Create synthetic step info from log
                    logger.debug(f"Using experiment.log for {task_name} (no step files)")
                    # For now, skip log-only parsing - focus on step_info files
                continue
            
            # Extract state-action pairs from each step
            for step_idx, step_info in enumerate(steps):
                state = self._extract_state_summary(step_info)
                action = self._extract_action(step_info)
                
                if not state or not action:
                    continue
                
                metadata = {
                    "task_name": task_name,
                    "step": step_idx,
                    "exp_dir": str(exp_dir),
                    "success": success,
                }
                
                if success:
                    successful_steps.append((step_info, metadata))
                else:
                    failed_steps.append((step_info, metadata))
        
        logger.info(f"Found {len(successful_steps)} successful steps and {len(failed_steps)} failed steps")
        
        # Create preference pairs
        examples = self._create_preference_pairs(successful_steps, failed_steps)
        
        logger.info(f"Generated {len(examples)} DPO examples")
        return examples
    
    def save_dataset(self, examples: List[DPOExample], output_file: Optional[Path] = None):
        """
        Save DPO dataset to JSONL file.
        
        Args:
            examples: List of DPOExample objects
            output_file: Optional output file path (defaults to self.output_file)
        """
        output_path = output_file or self.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + '\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    def generate_from_reflections(
        self,
        failure_reflections: List[Dict[str, Any]],
        successful_steps: List[Tuple[Any, Dict[str, Any]]],
    ) -> List[DPOExample]:
        """
        Generate DPO dataset using Claude Sonnet 3.7 thinking to reflect on failures
        and generate synthetic chosen responses.
        
        This method:
        1. Loads actual failed trajectories with real executed actions
        2. Uses Claude thinking to analyze why each failure occurred
        3. Uses Claude to suggest the correct real executable action (not abstract)
        4. Creates DPO pairs: rejected = actual failed action, chosen = Claude-suggested correct action
        
        Args:
            failure_reflections: List of failure reflection dicts from ReflectorAgent
            successful_steps: List of (step_info, metadata) tuples from successful tasks
            
        Returns:
            List of DPOExample objects with real executable actions only
        """
        if not self.claude_client:
            logger.warning("Claude client not initialized, falling back to direct generation")
            return self.generate_dataset()
        
        logger.info(f"Generating DPO dataset using Claude Sonnet 3.7 thinking on {len(failure_reflections)} failures")
        
        examples = []
        
        # Process each failure reflection
        for i, reflection in enumerate(failure_reflections):
            exp_dir = reflection.get("exp_dir")
            if not exp_dir:
                logger.debug(f"Skipping reflection {i}: no exp_dir")
                continue
            
            # Load actual trajectory from experiment
            trajectory = self._extract_trajectory(Path(exp_dir))
            if not trajectory:
                logger.warning(f"Skipping reflection {i}: could not extract trajectory from {exp_dir}")
                continue
            
            actions = trajectory.get("actions", [])
            logger.info(f"  Loaded trajectory with {len(actions)} actions")
            
            if not actions:
                logger.warning(f"Skipping reflection {i}: no actions in trajectory")
                continue
            
            # Extract full responses (reasoning + action) from log
            full_responses = self._extract_full_responses_from_log(Path(exp_dir))
            
            # Filter out noop responses - they're not useful for DPO training
            valid_responses = [r for r in full_responses if r and "noop()" not in r]
            
            if not valid_responses:
                logger.debug(f"Skipping reflection {i}: only noop responses in trajectory")
                continue
            
            # Generate multiple DPO examples per task (up to max_examples_per_task)
            # This increases dataset size and captures different failure points
            max_examples_per_task = 5  # Generate up to 5 examples per failed task
            responses_to_process = valid_responses[:max_examples_per_task]
            
            logger.info(f"  Processing {len(responses_to_process)} responses from this task")
            
            # Process each response to create multiple DPO examples
            for response_idx, failed_response in enumerate(responses_to_process):
            
                # Extract action from response for state matching
                action_match = re.search(r'```(\w+\([^`]*\))```', failed_response)
                failed_action = action_match.group(1) if action_match else ""
                
                # Try to get corresponding state and error information
                failed_state = ""
                action_error = ""
                all_actions = trajectory.get("actions", [])
                
                # Extract error for this specific action if available
                if trajectory.get("states") and len(trajectory["states"]) > 0:
                    # Match state index to action index if possible
                    try:
                        if failed_action:
                            action_idx = all_actions.index(failed_action) if failed_action in all_actions else response_idx
                        else:
                            action_idx = response_idx
                        if action_idx < len(trajectory["states"]):
                            failed_state = trajectory["states"][action_idx]
                            # Try to extract error from state if it contains error info
                            if "last_action_error" in failed_state or "error" in failed_state.lower():
                                error_match = re.search(r'(?:last_action_error|error)[:\s]+(.+?)(?:\n|$)', failed_state, re.IGNORECASE)
                                if error_match:
                                    action_error = error_match.group(1).strip()[:300]
                        else:
                            failed_state = trajectory["states"][min(response_idx, len(trajectory["states"])-1)]
                    except (ValueError, IndexError):
                        failed_state = trajectory["states"][min(response_idx, len(trajectory["states"])-1)] if trajectory["states"] else ""
                else:
                    # Create basic state from task name and error
                    task_name = reflection.get("task_name", "unknown")
                    error = reflection.get("error", "")
                    failed_state = f"Task: {task_name}"
                    if error:
                        failed_state += f"\nError: {error[:200]}"
                        action_error = error[:300]
                
                # If no action-specific error found, use task-level error
                if not action_error:
                    action_error = reflection.get("error", "")[:300]
                
                if not failed_state:
                    logger.debug(f"Skipping response {response_idx+1} from reflection {i}: no failed state")
                    continue
                
                if not failed_response:
                    logger.debug(f"Skipping response {response_idx+1} from reflection {i}: no failed response")
                    continue
                
                task_name = reflection.get('task_name', 'unknown')
                logger.info(f"  Processing response {response_idx+1}/{len(responses_to_process)} from {task_name}")
                logger.debug(f"    Failed response preview: {failed_response[:100]}...")
                logger.debug(f"    State length: {len(failed_state)} chars")
                
                # Use Claude to reflect on failure and suggest correct response
                dpo_example = self._generate_dpo_with_claude_reflection(
                    reflection=reflection,
                    failed_state=failed_state,
                    failed_response=failed_response,  # Pass full response, not just action
                    trajectory=trajectory,
                    action_error=action_error,  # Pass specific error for this action
                )
                
                if dpo_example:
                    examples.append(dpo_example)
                    logger.info(f"    ✓ Generated DPO example {len(examples)}: rejected={failed_response[:60]}..., chosen={dpo_example.chosen[:60]}...")
                else:
                    logger.warning(f"    ✗ Failed to generate DPO example for response {response_idx+1}")
                    logger.debug(f"       Failed response was: {failed_response[:200]}...")
                    logger.debug(f"       State preview: {failed_state[:100]}...")
        
        logger.info(f"Generated {len(examples)} DPO examples from real failures")
        
        # Generate synthetic examples to reach target size (50+)
        target_size = 50
        if len(examples) < target_size:
            logger.info(f"Generating {target_size - len(examples)} synthetic examples to reach target size of {target_size}")
            synthetic_examples = self._generate_synthetic_dpo_examples(
                failure_reflections=failure_reflections,
                existing_count=len(examples),
                target_count=target_size,
            )
            examples.extend(synthetic_examples)
            logger.info(f"Added {len(synthetic_examples)} synthetic examples. Total: {len(examples)}")
        
        logger.info(f"Generated {len(examples)} total DPO examples using Claude reflection")
        return examples
    
    def _generate_synthetic_dpo_examples(
        self,
        failure_reflections: List[Dict[str, Any]],
        existing_count: int,
        target_count: int,
    ) -> List[DPOExample]:
        """
        Generate synthetic DPO examples based on common failure patterns.
        
        Creates synthetic scenarios that encode generalizable strategies rather than
        task-specific details. This ensures the dataset teaches high-level policies.
        
        Args:
            failure_reflections: List of failure reflections to extract patterns from
            existing_count: Number of examples already generated
            target_count: Target total number of examples
            
        Returns:
            List of synthetic DPOExample objects
        """
        if not self.claude_client:
            logger.warning("Claude client not initialized, cannot generate synthetic examples")
            return []
        
        needed = max(0, target_count - existing_count)
        if needed == 0:
            return []
        
        logger.info(f"Generating {needed} synthetic DPO examples encoding generalizable strategies")
        
        # Extract common failure patterns from reflections
        failure_patterns = self._extract_failure_patterns(failure_reflections)
        
        synthetic_examples = []
        
        # Generate synthetic examples for each pattern
        examples_per_pattern = max(1, needed // max(len(failure_patterns), 1))
        
        for pattern_idx, pattern in enumerate(failure_patterns):
            if len(synthetic_examples) >= needed:
                break
            
            # Generate multiple variations of this pattern
            for var_idx in range(examples_per_pattern):
                if len(synthetic_examples) >= needed:
                    break
                
                synthetic_example = self._generate_synthetic_example_for_pattern(pattern, var_idx)
                if synthetic_example:
                    synthetic_examples.append(synthetic_example)
                    logger.debug(f"  Generated synthetic example {len(synthetic_examples)}/{needed} for pattern: {pattern['type']}")
        
        logger.info(f"Generated {len(synthetic_examples)} synthetic DPO examples")
        return synthetic_examples
    
    def _extract_failure_patterns(self, failure_reflections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract common failure patterns from reflections.
        
        Returns list of pattern dicts with type, error, and generalizable strategy.
        """
        patterns = []
        
        # Common failure pattern types
        pattern_types = {
            "display_task_missing_communication": {
                "type": "display_task_missing_communication",
                "error": "Task did not pass evaluation criteria - agent did not communicate findings",
                "reflection": "When task goal contains 'display', 'show', 'find', or 'report', agent must use send_msg_to_user() to communicate information",
                "generalizable_strategy": "Always use send_msg_to_user() when task requires communicating information to the user",
            },
            "wrong_element_type": {
                "type": "wrong_element_type",
                "error": "Element is not a <select> element",
                "reflection": "Agent tried to use select_option() on non-select element. Must check element type before using actions",
                "generalizable_strategy": "Verify element type matches action type before executing (select_option for <select>, fill for <input>, click for buttons/links)",
            },
            "multi_action_error": {
                "type": "multi_action_error",
                "error": "Received a multi-action, only single-actions are allowed",
                "reflection": "Agent attempted multiple actions in one turn. Only one action per step is allowed",
                "generalizable_strategy": "Execute only one action per turn. Break complex operations into individual steps",
            },
            "incomplete_task": {
                "type": "incomplete_task",
                "error": "Task did not pass evaluation criteria - incomplete execution",
                "reflection": "Agent started task but did not complete all required steps. Must follow through on entire task sequence",
                "generalizable_strategy": "Complete all required steps in a task. Don't stop after partial progress",
            },
        }
        
        # Check which patterns appear in actual failures
        for reflection in failure_reflections:
            error = reflection.get("error", "").lower()
            task_goal = reflection.get("task_name", "").lower()
            
            if any(kw in task_goal for kw in ["display", "show", "find", "report"]) or "communicat" in error:
                if "display_task_missing_communication" not in [p["type"] for p in patterns]:
                    patterns.append(pattern_types["display_task_missing_communication"])
            
            if "select_option" in error or "not a <select>" in error:
                if "wrong_element_type" not in [p["type"] for p in patterns]:
                    patterns.append(pattern_types["wrong_element_type"])
            
            if "multi-action" in error or "multiple actions" in error:
                if "multi_action_error" not in [p["type"] for p in patterns]:
                    patterns.append(pattern_types["multi_action_error"])
        
        # Always include core patterns if not found
        core_patterns = ["display_task_missing_communication", "wrong_element_type", "multi_action_error"]
        for pattern_type in core_patterns:
            if pattern_type not in [p["type"] for p in patterns]:
                patterns.append(pattern_types[pattern_type])
        
        return patterns
    
    def _generate_synthetic_example_for_pattern(
        self,
        pattern: Dict[str, Any],
        variation_idx: int,
    ) -> Optional[DPOExample]:
        """
        Generate a synthetic DPO example for a specific failure pattern.
        
        Creates a synthetic scenario that teaches the generalizable strategy.
        """
        if not self.claude_client:
            return None
        
        # Build synthetic prompt based on pattern
        prompt = self._build_synthetic_dpo_prompt(pattern, variation_idx)
        
        try:
            # Configure thinking
            if self.use_thinking:
                thinking_budget = 10000
                max_tokens = 12000
                thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}
            else:
                max_tokens = 4000
                thinking_config = {"type": "disabled"}
            
            # Query Claude
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=max_tokens,
                thinking=thinking_config,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            
            # Extract text content
            text_content = None
            for content_block in response.content:
                if content_block.type == "text":
                    text_content = content_block.text
                    break
            
            if not text_content:
                return None
            
            # Parse response - should contain both rejected and chosen
            parsed = self._parse_synthetic_dpo_response(text_content, pattern)
            if parsed:
                return DPOExample(
                    prompt=parsed["prompt"],
                    chosen=parsed["chosen"],
                    rejected=parsed["rejected"],
                    metadata={"synthetic": True, "pattern": pattern["type"]}
                )
            
        except Exception as e:
            logger.warning(f"Failed to generate synthetic example for pattern {pattern['type']}: {e}")
            return None
        
        return None
    
    def _build_synthetic_dpo_prompt(
        self,
        pattern: Dict[str, Any],
        variation_idx: int,
    ) -> str:
        """
        Build prompt for Claude to generate a synthetic DPO example that encodes a generalizable strategy.
        """
        action_set_desc = self._get_action_set_description()
        
        # Create synthetic scenario based on pattern type
        if pattern["type"] == "display_task_missing_communication":
            synthetic_state = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nFind and display information about [a product/item/information] on the page.\n\n⚠️ IMPORTANT: This task requires COMMUNICATING information to the user using send_msg_to_user()"
            synthetic_rejected = "I found the information on the page. The task is complete.\n\n```noop()```"
            strategy_focus = "When task goal contains 'display', 'show', 'find', or 'report', you MUST use send_msg_to_user() to communicate the information. Simply seeing it on the page is not enough."
        
        elif pattern["type"] == "wrong_element_type":
            synthetic_state = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nChange a quantity or select an option on the page.\n\nCurrent page has a quantity selector or option selector element."
            synthetic_rejected = "I need to change the quantity. I'll use select_option() on the quantity element.\n\n```select_option(\"123\", \"5\")```"
            strategy_focus = "Before using select_option(), verify the element is actually a <select> element. Many quantity selectors are input fields that require fill() instead. Check element type before choosing action."
        
        elif pattern["type"] == "multi_action_error":
            synthetic_state = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nComplete a multi-step task on the page.\n\nCurrent page state with multiple elements available."
            synthetic_rejected = "I need to fill the form and click submit. I'll do both actions now.\n\n```fill(\"123\", \"text\")```\n```click(\"456\")```"
            strategy_focus = "Execute only ONE action per turn. Break complex operations into individual steps. Wait for environment response before proceeding."
        
        elif pattern["type"] == "incomplete_task":
            synthetic_state = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nComplete a multi-step process: [step 1], [step 2], and [step 3].\n\nCurrent page shows progress on step 1."
            synthetic_rejected = "I've completed step 1. The task is done.\n\n```noop()```"
            strategy_focus = "Complete ALL required steps in a task. Don't stop after partial progress. Review the full task goal to ensure all steps are completed."
        
        else:
            # Generic pattern
            synthetic_state = f"## TASK GOAL (CRITICAL - READ CAREFULLY):\nComplete a task on the page.\n\nError context: {pattern.get('error', 'Unknown error')}"
            synthetic_rejected = "I'll proceed with the task.\n\n```noop()```"
            strategy_focus = pattern.get("generalizable_strategy", "Apply correct strategy based on task requirements")
        
        prompt_parts = [
            "You are generating a synthetic DPO training example that teaches a GENERALIZABLE STRATEGY.",
            "",
            "## OBJECTIVE:",
            f"Create a DPO example that encodes this generalizable strategy:",
            f"  {strategy_focus}",
            "",
            "## SYNTHETIC SCENARIO:",
            synthetic_state,
            "",
            "## REJECTED RESPONSE (what demonstrates the failure pattern):",
            synthetic_rejected,
            "",
            "## AVAILABLE ACTIONS:",
            action_set_desc,
            "",
            "## YOUR TASK:",
            "",
            "Generate a CHOSEN response that demonstrates the CORRECT generalizable strategy.",
            "",
            "CRITICAL REQUIREMENTS:",
            "1. The chosen response must encode the generalizable strategy, not task-specific details",
            "2. Write as the AGENT would respond (1-3 sentences reasoning + action code block)",
            "3. Do NOT include analysis or explanations - just the agent's natural response",
            "4. The reasoning should demonstrate understanding of the generalizable strategy",
            "5. Use real executable actions from the available actions list",
            "6. Make it generalizable - avoid specific product names, bids, or task details",
            "",
            "Return your response in this format:",
            "CHOSEN:",
            "[agent's reasoning - 1-3 sentences]",
            "",
            "```action(\"args\")```",
            "",
            "The chosen response should teach the model the correct generalizable strategy.",
        ]
        
        return "\n".join(prompt_parts)
    
    def _parse_synthetic_dpo_response(self, response: str, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Parse Claude's response to extract synthetic DPO example components.
        
        Expected format:
        CHOSEN:
        [reasoning]
        
        ```action("args")```
        """
        if not response:
            return None
        
        # Extract chosen response
        chosen_match = re.search(r'CHOSEN:\s*(.+?)(?=\n\n```|$)', response, re.DOTALL)
        if not chosen_match:
            # Try without CHOSEN: prefix
            chosen_match = re.search(r'(.+?)(?=\n\n```|$)', response, re.DOTALL)
        
        if not chosen_match:
            return None
        
        chosen_reasoning = chosen_match.group(1).strip()
        
        # Extract action code block
        action_match = re.search(r'```(\w+\([^`]*\))```', response)
        if not action_match:
            return None
        
        chosen_action = action_match.group(1)
        if not self._is_valid_executable_action(chosen_action):
            return None
        
        # Combine into full chosen response
        chosen_response = f"{chosen_reasoning}\n\n```{chosen_action}```"
        
        # Create synthetic rejected response based on pattern
        if pattern["type"] == "display_task_missing_communication":
            rejected_response = "I found the information on the page. The task is complete.\n\n```noop()```"
            prompt = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nFind and display information about an item on the page.\n\n⚠️ IMPORTANT: This task requires COMMUNICATING information to the user using send_msg_to_user()"
        elif pattern["type"] == "wrong_element_type":
            rejected_response = "I'll use select_option() to change the value.\n\n```select_option(\"123\", \"5\")```"
            prompt = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nChange a quantity or select an option on the page.\n\nCurrent page has a quantity selector element."
        elif pattern["type"] == "multi_action_error":
            rejected_response = "I need to fill the form and click submit.\n\n```fill(\"123\", \"text\")```\n```click(\"456\")```"
            prompt = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nComplete a multi-step task on the page.\n\nCurrent page state with multiple elements available."
        elif pattern["type"] == "incomplete_task":
            rejected_response = "I've completed the first step. The task is done.\n\n```noop()```"
            prompt = "## TASK GOAL (CRITICAL - READ CAREFULLY):\nComplete a multi-step process: step 1, step 2, and step 3.\n\nCurrent page shows progress on step 1."
        else:
            rejected_response = "I'll proceed with the task.\n\n```noop()```"
            prompt = f"## TASK GOAL (CRITICAL - READ CAREFULLY):\nComplete a task on the page.\n\nError context: {pattern.get('error', 'Unknown error')}"
        
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
    
    def _generate_dpo_with_claude_reflection(
        self,
        reflection: Dict[str, Any],
        failed_state: str,
        failed_response: str,  # Changed from failed_action - now full response
        trajectory: Dict[str, Any],
        action_error: str = "",  # Specific error for this action
    ) -> Optional[DPOExample]:
        """
        Use Claude Sonnet 3.7 thinking to reflect on failure and suggest correct response.
        
        Returns a DPOExample where:
        - rejected: The actual failed response (reasoning + action)
        - chosen: Claude-suggested correct response (reasoning + action)
        """
        if not self.claude_client:
            return None
        
        # Build prompt for Claude reflection
        prompt = self._build_reflection_prompt_for_dpo(reflection, failed_state, failed_response, trajectory, action_error)
        
        try:
            # Configure thinking
            if self.use_thinking:
                thinking_budget = 10000
                max_tokens = 12000
                thinking_config = {"type": "enabled", "budget_tokens": thinking_budget}
            else:
                max_tokens = 4000
                thinking_config = {"type": "disabled"}
            
            # Query Claude
            response = self.claude_client.messages.create(
                model=self.claude_model,
                max_tokens=max_tokens,
                thinking=thinking_config,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            
            # Extract text content
            text_content = None
            for content_block in response.content:
                if content_block.type == "text":
                    text_content = content_block.text
                    break
            
            if not text_content:
                logger.warning("No text content in Claude response")
                return None
            
            # Parse Claude's suggested response (should be reasoning + action code block)
            suggested_response = self._parse_claude_suggested_response(text_content, failed_response)
            
            if not suggested_response:
                logger.warning(f"Could not parse valid response from Claude: {text_content[:200]}")
                return None
            
            logger.debug(f"Claude suggested response: {suggested_response[:200]}...")
            
            # Generalize the chosen response to remove task-specific details
            generalized_chosen = self._generalize_response(suggested_response, reflection.get("task_name", ""))
            generalized_state = self._generalize_state(failed_state, reflection.get("task_name", ""))
            generalized_rejected = self._generalize_response(failed_response, reflection.get("task_name", ""))
            
            # Create DPO example with full responses (generalized)
            return DPOExample(
                prompt=generalized_state,
                chosen=generalized_chosen,  # Claude-suggested correct complete response (generalized)
                rejected=generalized_rejected,  # Actual failed complete response (generalized)
                metadata=None
            )
            
        except Exception as e:
            logger.error(f"Failed to generate DPO example with Claude reflection: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _build_reflection_prompt_for_dpo(
        self,
        reflection: Dict[str, Any],
        failed_state: str,
        failed_response: str,  # Changed from failed_action - now full response
        trajectory: Dict[str, Any],
        action_error: str = "",  # Specific error for this action
    ) -> str:
        """Build prompt for Claude to reflect on failure and suggest correct response."""
        
        task_name = reflection.get("task_name", "unknown")
        
        # Load task config to get goal and evaluation rubric
        task_config = self._load_task_config(task_name)
        task_goal = ""
        eval_rubric = ""
        
        if task_config:
            task_goal = task_config.get("goal", "")
            # Extract evaluation rubric (critical for understanding what "display" means)
            evals = task_config.get("evals", [])
            if evals:
                rubric_parts = []
                for eval_config in evals:
                    if eval_config.get("rubric"):
                        rubric_parts.append(f"- {eval_config['rubric']}")
                    elif eval_config.get("description"):
                        rubric_parts.append(f"- {eval_config['description']}")
                if rubric_parts:
                    eval_rubric = "\n".join(rubric_parts)
        
        # Get all available actions
        action_set_desc = self._get_action_set_description()
        
        # Build comprehensive prompt
        prompt_parts = [
            "You are analyzing a failed task execution to determine what the correct response should have been.",
            "",
            "## TASK DEFINITION:",
            f"Task Name: {task_name}",
        ]
        
        if task_goal:
            prompt_parts.append(f"Task Goal: {task_goal}")
            prompt_parts.append("")
            prompt_parts.append("CRITICAL: Pay special attention to keywords in the goal:")
            prompt_parts.append("- If goal says 'display', 'show', 'find', or 'report' → you MUST use send_msg_to_user()")
            prompt_parts.append("- If goal says 'search' → you need to fill search box and click search button")
            prompt_parts.append("- The goal tells you what the task requires")
            prompt_parts.append("")
        
        if eval_rubric:
            prompt_parts.append("## EVALUATION CRITERIA (How success is measured):")
            prompt_parts.append(eval_rubric)
            prompt_parts.append("")
            prompt_parts.append("IMPORTANT: The evaluation checks if the agent's RESPONSE mentions the required information.")
            prompt_parts.append("If the task asks to 'display' something, the agent must use send_msg_to_user() to communicate it.")
            prompt_parts.append("Simply seeing information on the page is NOT enough - it must be communicated to the user.")
            prompt_parts.append("")
        
        prompt_parts.extend([
            "## AVAILABLE ACTIONS:",
            action_set_desc,
            "",
            "## TASK CONTEXT:",
            f"Error: {reflection.get('error', 'Unknown error')}",
            f"Reflection: {reflection.get('reflection', '')}",
            "",
            "## STATE (what the agent observed):",
            "NOTE: The state below may contain task-specific details. Focus on the GENERALIZABLE STRATEGY needed, not the specific details.",
            failed_state[:1000],  # Increased limit to include more context
            "",
            "## REJECTED RESPONSE (actual failed response from the environment):",
            "This is the ACTUAL response the agent gave that led to failure. Analyze WHY it failed.",
            failed_response,
            "",
        ])
        
        # Add action-specific error if available
        if action_error:
            prompt_parts.extend([
                "## ACTION ERROR (what went wrong with this specific action):",
                f"{action_error}",
                "",
                "This error message tells you WHY the rejected response failed. Use this to identify the root cause.",
                "",
            ])
        
        prompt_parts.extend([
            "## TRAJECTORY CONTEXT:",
            f"- Total steps: {trajectory.get('n_steps', 0)}",
            f"- Invalid actions: {trajectory.get('invalid_actions', 0)}",
            f"- Recent actions: {', '.join(trajectory.get('actions', [])[:5])}",
            "",
            "## STEP 1: ANALYZE WHY THE REJECTED RESPONSE FAILED",
            "",
            "Before generating the correct response, you must FIRST analyze WHY the rejected response failed.",
            "Think about the ROOT CAUSE - what generalizable principle was violated?",
            "",
            "Use the error message (if provided) and the rejected response to identify:",
            "1. What specific mistake was made?",
            "2. What generalizable principle was violated?",
            "3. What strategy would have prevented this failure?",
            "",
            "Common failure patterns to identify:",
            "- Wrong action type (e.g., used select_option() on input field, or fill() on dropdown) → Principle: Check element type before choosing action",
            "- Multi-action error (tried to execute multiple actions in one turn) → Principle: Execute one action at a time",
            "- Missing communication (task requires send_msg_to_user() but agent didn't use it) → Principle: Use send_msg_to_user() when task requires communication",
            "- Assumed state (agent assumed page state without checking) → Principle: Verify current state before proceeding",
            "- Wrong sequencing (agent skipped steps or did them out of order) → Principle: Complete steps in proper sequence",
            "- Element type mismatch (agent didn't check element type before choosing action) → Principle: Verify element properties before action selection",
            "",
            "Identify the SPECIFIC generalizable principle that was violated, then generate a response that explicitly demonstrates the correct principle.",
            "",
            "## STEP 2: GENERATE GENERALIZABLE CORRECT RESPONSE",
            "",
            "Now generate the CORRECT response that the agent should have given.",
            "This response must:",
            "1. Explicitly encode the generalizable strategy that would have prevented the failure",
            "2. Demonstrate the correct generalizable principle in the reasoning",
            "3. Be written as the AGENT would respond (not as an analyst)",
            "",
            "CRITICAL: This response will be used for DPO training to teach GENERALIZABLE STRATEGIES.",
            "Your response must encode high-level policies that apply across different tasks, not task-specific details.",
            "",
            "IMPORTANT: Write as if YOU are the agent making the correct decision in that moment.",
            "Do NOT write analysis about the failure - instead, demonstrate the correct generalizable strategy in your reasoning.",
            "",
            "The agent's response format is:",
            "1. Brief reasoning text (1-3 sentences) explaining what you're doing and why",
            "2. A code block with the action: ```action(\"args\")```",
            "",
            "CRITICAL REQUIREMENTS FOR GENERALIZABILITY:",
            "1. Write as the AGENT would respond, not as an analyst",
            "2. Keep reasoning concise and focused on the current step (1-3 sentences max)",
            "3. EXPLICITLY state the generalizable principle in your reasoning (e.g., 'I need to check element type before choosing action')",
            "4. ENCODE GENERALIZABLE STRATEGIES in your reasoning - focus on policies that apply across tasks",
            "5. AVOID task-specific details (product names, specific bids, task names) - use generic placeholders",
            "6. The action MUST be a real executable action from the available actions list above",
            "7. It must be a SINGLE action, not multiple actions or a sequence",
            "8. If the task goal contains 'display', 'show', 'find', or 'report', you MUST use send_msg_to_user()",
            "9. Pay attention to the evaluation criteria - what does it check for?",
            "10. Focus on high-level decision-making patterns, not task-specific details",
            "11. Your reasoning should demonstrate WHY your approach is correct (the generalizable principle)",
            "",
            "GENERALIZABILITY EXAMPLES (with explicit principles):",
            "",
            "BAD (too specific, no principle):",
            "I need to search for 'Marshall Emberton II Portable Bluetooth Speaker' using bid '162'.",
            "",
            "GOOD (generalizable, explicit principle):",
            "I need to search for the item. I'll fill the search box with the product name, then click the search button.",
            "",
            "BAD (too specific, implicit principle):",
            "The task asks me to display the VILVA Portable-Monitor-for-Laptop.",
            "",
            "GOOD (generalizable, explicit communication principle):",
            "I've found the first product in the search results. The task requires me to display information, which means I must communicate it to the user using send_msg_to_user(). Simply seeing it on the page is not enough.",
            "",
            "GOOD EXAMPLE (element type checking - explicit principle):",
            "I need to change the quantity, but first I should check what type of element this is. Many quantity selectors look like dropdowns but are actually input fields. I'll verify the element type before choosing the action.",
            "",
            "```fill(\"[bid]\", \"[quantity]\")```",
            "",
            "GOOD EXAMPLE (multi-action prevention - explicit principle):",
            "I'll execute one action at a time and wait for the page response before proceeding. This ensures each step completes properly before moving to the next.",
            "",
            "```fill(\"[bid]\", \"[text]\")```",
            "",
            "BAD EXAMPLES (don't do these):",
            "# Analysis of the Failed Response... (too much analysis)",
            "I need to search for 'Marshall Emberton II' using bid '162' (too specific)",
            "The task v1.omnizon-4 requires... (mentions specific task)",
            "I'll fill the search box. (no generalizable principle stated)",
            "",
            "Remember:",
            "- Encode GENERALIZABLE STRATEGIES, not task-specific details",
            "- EXPLICITLY state the generalizable principle in your reasoning",
            "- Use generic placeholders for specific values (product names, bids, etc.)",
            "- Your reasoning should teach the model WHY your approach is correct",
            "",
            "Return your response now (as the agent would respond, encoding generalizable strategies with explicit principles):",
            "",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_claude_suggested_response(self, response: str, fallback_response: str) -> Optional[str]:
        """
        Parse Claude's response to extract the complete suggested response (reasoning + action).
        
        Returns the full response string in format: "reasoning\n\n```action(\"args\")```" if valid, None otherwise.
        """
        if not response:
            logger.warning("Empty response from Claude")
            return None
        
        # Clean up response - remove extra whitespace
        response = response.strip()
        
        # Look for action code block: ```action("args")```
        action_code_block_pattern = r'```(\w+\([^`]*\))```'
        action_match = re.search(action_code_block_pattern, response)
        
        if action_match:
            action = action_match.group(1)
            
            # Validate it's a real executable action
            if not self._is_valid_executable_action(action):
                logger.warning(f"Invalid action in Claude response: {action}")
                return None
            
            # Extract reasoning text before the code block
            reasoning = response[:action_match.start()].strip()
            
            # Combine into full response format
            if reasoning:
                full_response = f"{reasoning}\n\n```{action}```"
            else:
                # If no reasoning, just use the action code block
                full_response = f"```{action}```"
            
            logger.debug(f"Parsed full response: reasoning={len(reasoning)} chars, action={action}")
            return full_response
        
        # Fallback: try to find action without code blocks
        action_patterns = [
            r'(click\("[^"]+"\))',
            r'(fill\("[^"]+",\s*"[^"]*"\))',
            r'(scroll\("?[^"]*"?\))',
            r'(send_msg_to_user\("[^"]*"\))',
            r'(select_option\("[^"]+",\s*"[^"]*"\))',
            r'(focus\("[^"]+"\))',
            r'(goto\("[^"]+"\))',
            r'(noop\(\))',
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, response)
            if match:
                action = match.group(1)
                if self._is_valid_executable_action(action):
                    # Extract text before action as reasoning
                    reasoning = response[:match.start()].strip()
                    if reasoning:
                        full_response = f"{reasoning}\n\n```{action}```"
                    else:
                        full_response = f"```{action}```"
                    logger.debug(f"Found action via pattern (no code block): {action}")
                    return full_response
        
        logger.warning(f"Could not parse valid response from Claude: {response[:200]}...")
        return None
    
    def _parse_claude_suggested_action(self, response: str, fallback_action: str) -> Optional[str]:
        """
        Parse Claude's response to extract the suggested real executable action.
        
        Returns the action string if valid, None otherwise.
        DEPRECATED: Use _parse_claude_suggested_response for full responses.
        """
        if not response:
            logger.warning("Empty response from Claude")
            return None
        
        # Clean up response - remove extra whitespace
        response = response.strip()
        
        # Try to extract action from response
        # Look for patterns like: click("123"), fill("456", "text"), etc.
        action_patterns = [
            r'(click\("[^"]+"\))',
            r'(fill\("[^"]+",\s*"[^"]*"\))',
            r'(scroll\("?[^"]*"?\))',
            r'(send_msg_to_user\("[^"]*"\))',
            r'(select_option\("[^"]+",\s*"[^"]*"\))',
            r'(focus\("[^"]+"\))',
            r'(goto\("[^"]+"\))',
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, response)
            if match:
                action = match.group(1)
                # Validate it's a real executable action
                if self._is_valid_executable_action(action):
                    logger.debug(f"Found action via pattern {pattern}: {action}")
                    return action
        
        # If no pattern matched, try to extract first line that looks like an action
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Remove markdown code blocks if present
            line = re.sub(r'^```\w*', '', line)
            line = re.sub(r'```$', '', line)
            line = line.strip()
            
            # Remove quotes if the entire line is quoted
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            if self._is_valid_executable_action(line):
                logger.debug(f"Found action in line: {line}")
                return line
        
        # Last resort: look for any action-like pattern in the response
        all_matches = []
        for pattern in action_patterns:
            matches = re.findall(pattern, response)
            all_matches.extend(matches)
        
        if all_matches:
            # Return the first valid action found
            for match in all_matches:
                if self._is_valid_executable_action(match):
                    logger.debug(f"Found action in all matches: {match}")
                    return match
        
        # Log the full response for debugging
        logger.warning(f"Could not parse valid action from Claude response.")
        logger.warning(f"Response (first 500 chars): {response[:500]}")
        logger.warning(f"Response (full): {response}")
        return None
    
    def _is_valid_executable_action(self, action: str) -> bool:
        """Check if action is a valid real executable action."""
        if not isinstance(action, str):
            return False
        
        action = action.strip()
        
        # Must start with a valid action type
        valid_prefixes = [
            'click(',
            'fill(',
            'scroll(',
            'send_msg_to_user(',
            'select_option(',
            'focus(',
            'goto(',
            'noop(',  # noop is a valid action (though not very useful for DPO)
        ]
        
        return any(action.startswith(prefix) for prefix in valid_prefixes)
    
    def _generalize_response(self, response: str, task_name: str) -> str:
        """
        Generalize a response by replacing task-specific details with generic placeholders.
        
        This ensures the DPO dataset teaches generalizable strategies, not task-specific details.
        """
        if not response:
            return response
        
        generalized = response
        
        # Remove task name references
        if task_name:
            generalized = generalized.replace(task_name, "[task]")
            # Also remove version prefixes
            generalized = re.sub(r'v\d+\.', '', generalized)
        
        # Replace specific product names with generic placeholders
        # Common product name patterns
        product_patterns = [
            r'"[^"]*Portable[^"]*"',  # Product names with "Portable"
            r'"[^"]*SAMSUNG[^"]*"',  # Samsung products
            r'"[^"]*Marshall[^"]*"',  # Marshall products
            r'"[^"]*PlayStation[^"]*"',  # PlayStation products
            r'"[^"]*Michael Kors[^"]*"',  # Michael Kors products
        ]
        
        for pattern in product_patterns:
            generalized = re.sub(pattern, '"[product name]"', generalized, flags=re.IGNORECASE)
        
        # Replace specific bids with generic placeholders (but keep the action structure)
        # Pattern: action("specific_bid") -> action("[bid]")
        generalized = re.sub(r'(\w+\(")(\d+)(")', r'\1[bid]\3', generalized)
        
        # Replace specific product details in send_msg_to_user with generic placeholders
        # Pattern: send_msg_to_user("very long specific product description...")
        # -> send_msg_to_user("[product information]")
        msg_pattern = r'send_msg_to_user\("([^"]{50,})"\)'
        generalized = re.sub(msg_pattern, 'send_msg_to_user("[product information]")', generalized)
        
        # Remove specific task references in reasoning
        generalized = re.sub(r'task (v\d+\.)?\w+-\d+', 'the task', generalized, flags=re.IGNORECASE)
        generalized = re.sub(r'omnizon', '[website]', generalized, flags=re.IGNORECASE)
        
        return generalized
    
    def _generalize_state(self, state: str, task_name: str) -> str:
        """
        Generalize a state description by replacing task-specific details.
        """
        if not state:
            return state
        
        generalized = state
        
        # Remove task name
        if task_name:
            generalized = generalized.replace(task_name, "[task]")
            generalized = re.sub(r'v\d+\.', '', generalized)
        
        # Replace specific product names
        generalized = re.sub(r'"[^"]*Portable[^"]*"', '"[product]"', generalized, flags=re.IGNORECASE)
        generalized = re.sub(r'"[^"]*SAMSUNG[^"]*"', '"[product]"', generalized, flags=re.IGNORECASE)
        generalized = re.sub(r'"[^"]*Marshall[^"]*"', '"[product]"', generalized, flags=re.IGNORECASE)
        
        # Replace specific bids in state descriptions
        generalized = re.sub(r'bid [\'"]?\d+[\'"]?', 'bid [bid]', generalized)
        generalized = re.sub(r'\[(\d+)\]', '[bid]', generalized)
        
        # Keep the task goal structure but abstract specific details
        # The goal itself is important context, but we can note it's a placeholder
        
        return generalized
    
    def generate_and_save(self) -> Path:
        """
        Generate dataset and save to file.
        
        Returns:
            Path to saved file
        """
        examples = self.generate_dataset()
        self.save_dataset(examples)
        return self.output_file
    
    def generate_and_save_from_reflections(
        self,
        failure_reflections: List[Dict[str, Any]],
        successful_steps: List[Tuple[Any, Dict[str, Any]]],
    ) -> Path:
        """
        Generate DPO dataset from actual trajectories and save to file.
        
        Args:
            failure_reflections: List of failure reflection dicts (used to identify failed experiments)
            successful_steps: List of (step_info, metadata) tuples from successful tasks
            
        Returns:
            Path to saved file
        """
        examples = self.generate_from_reflections(failure_reflections, successful_steps)
        self.save_dataset(examples)
        return self.output_file


def main():
    """CLI entry point for generating DPO datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate DPO dataset from experiment logs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./dpo_dataset.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum number of steps required"
    )
    parser.add_argument(
        "--include-failed-only",
        action="store_true",
        help="Only include examples from failed tasks"
    )
    
    args = parser.parse_args()
    
    generator = DPODatasetGenerator(
        results_dir=args.results_dir,
        output_file=args.output_file,
        min_steps=args.min_steps,
        include_failed_only=args.include_failed_only,
    )
    
    output_path = generator.generate_and_save()
    print(f"✅ DPO dataset saved to: {output_path}")


if __name__ == "__main__":
    import os
    main()

