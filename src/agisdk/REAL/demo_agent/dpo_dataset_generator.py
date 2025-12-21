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
        except Exception as e:
            logger.warning(f"Failed to initialize Claude client: {e}")
            self.claude_client = None
    
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
        
        # Goal/instruction
        if obs.get("goal_object"):
            goal = obs["goal_object"]
            if isinstance(goal, list) and len(goal) > 0:
                if isinstance(goal[0], dict) and "text" in goal[0]:
                    state_parts.append(f"Goal: {goal[0]['text']}")
            elif isinstance(goal, str):
                state_parts.append(f"Goal: {goal}")
        
        # URL
        if obs.get("url"):
            state_parts.append(f"URL: {obs['url']}")
        
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
                                    base_state += f"\nGoal: {goal_text[:500]}"
                            elif isinstance(goal_object, str):
                                base_state += f"\nGoal: {goal_object[:500]}"
                    except Exception as e:
                        logger.debug(f"Could not load goal_object: {e}")
                
                # Also try to extract goal from experiment.log
                log_path = exp_dir / "experiment.log"
                if log_path.exists() and "Goal:" not in base_state:
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
                                    base_state += f"\nGoal: {goal_text}"
                                    break
                    except Exception as e:
                        logger.debug(f"Could not extract goal from log: {e}")
                
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
            
            # Get the state and action where failure occurred
            # Filter out noop actions - they're not useful for DPO training
            all_actions = trajectory.get("actions", [])
            valid_actions = [a for a in all_actions if a and not a.startswith("noop(")]
            
            if not valid_actions:
                logger.debug(f"Skipping reflection {i}: only noop actions in trajectory")
                continue
            
            # Use the first meaningful action (not noop)
            failed_action = valid_actions[0]
            
            # Try to get corresponding state
            failed_state = ""
            if trajectory.get("states") and len(trajectory["states"]) > 0:
                # Match state index to action index if possible
                try:
                    action_idx = all_actions.index(failed_action)
                    if action_idx < len(trajectory["states"]):
                        failed_state = trajectory["states"][action_idx]
                    else:
                        failed_state = trajectory["states"][0]
                except (ValueError, IndexError):
                    failed_state = trajectory["states"][0]
            else:
                # Create basic state from task name and error
                task_name = reflection.get("task_name", "unknown")
                error = reflection.get("error", "")
                failed_state = f"Task: {task_name}"
                if error:
                    failed_state += f"\nError: {error[:200]}"
            
            if not failed_state:
                logger.debug(f"Skipping reflection {i}: no failed state")
                continue
            
            if not failed_action:
                logger.debug(f"Skipping reflection {i}: no failed action")
                continue
            
            task_name = reflection.get('task_name', 'unknown')
            logger.info(f"Processing failure {i+1}/{len(failure_reflections)}: {task_name}")
            logger.info(f"  Failed action: {failed_action}")
            logger.info(f"  State length: {len(failed_state)} chars")
            
            # Use Claude to reflect on failure and suggest correct action
            dpo_example = self._generate_dpo_with_claude_reflection(
                reflection=reflection,
                failed_state=failed_state,
                failed_action=failed_action,
                trajectory=trajectory,
            )
            
            if dpo_example:
                examples.append(dpo_example)
                logger.info(f"  ✓ Generated DPO example: rejected={failed_action[:50]}..., chosen={dpo_example.chosen[:50]}...")
            else:
                logger.warning(f"  ✗ Failed to generate DPO example for {task_name}")
                logger.debug(f"     Failed action was: {failed_action}")
                logger.debug(f"     State preview: {failed_state[:100]}...")
        
        logger.info(f"Generated {len(examples)} DPO examples using Claude reflection")
        return examples
    
    def _generate_dpo_with_claude_reflection(
        self,
        reflection: Dict[str, Any],
        failed_state: str,
        failed_action: str,
        trajectory: Dict[str, Any],
    ) -> Optional[DPOExample]:
        """
        Use Claude Sonnet 3.7 thinking to reflect on failure and suggest correct action.
        
        Returns a DPOExample where:
        - rejected: The actual failed action (real executable action)
        - chosen: Claude-suggested correct action (also a real executable action, not abstract)
        """
        if not self.claude_client:
            return None
        
        # Build prompt for Claude reflection
        prompt = self._build_reflection_prompt_for_dpo(reflection, failed_state, failed_action, trajectory)
        
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
            
            # Parse Claude's suggested action
            suggested_action = self._parse_claude_suggested_action(text_content, failed_action)
            
            if not suggested_action:
                logger.warning(f"Could not parse valid action from Claude response: {text_content[:200]}")
                return None
            
            logger.debug(f"Claude suggested action: {suggested_action}")
            
            # Create DPO example with real executable actions
            return DPOExample(
                prompt=failed_state,
                chosen=suggested_action,  # Claude-suggested correct real executable action
                rejected=failed_action,  # Actual failed real executable action
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
        failed_action: str,
        trajectory: Dict[str, Any],
    ) -> str:
        """Build prompt for Claude to reflect on failure and suggest correct action."""
        prompt = """You are analyzing a failed task execution to determine what the correct action should have been.

## TASK CONTEXT:
Task: {task_name}
Error: {error}
Reflection: {reflection}

## STATE (what the agent observed):
{state}

## FAILED ACTION (what the agent actually did):
{action}

## TRAJECTORY CONTEXT:
- Total steps: {n_steps}
- Invalid actions: {invalid_actions}
- All actions taken: {all_actions}

## YOUR TASK:

Analyze why this action failed and suggest the CORRECT real executable action that should have been taken instead.

CRITICAL REQUIREMENTS:
1. The suggested action MUST be a real executable action (e.g., click("123"), fill("456", "text"), scroll(), send_msg_to_user("message"))
2. It must be a SINGLE action, not multiple actions or a sequence
3. It must be concrete and executable, NOT abstract reasoning or planning
4. It should address the specific error that occurred

Valid action formats:
- click("bid")
- fill("bid", "text")
- scroll("up")
- send_msg_to_user("message")
- select_option("bid", "option")
- focus("bid")
- goto("url")

IMPORTANT: Return ONLY the action string itself, nothing else. No explanations, no JSON, no markdown, just the action.
For example, if the correct action is to click button 123, return exactly: click("123")

If you need to fill a field, return: fill("456", "text")
If you need to scroll, return: scroll("down")
If you need to send a message, return: send_msg_to_user("message")

Return your response now:

""".format(
            task_name=reflection.get("task_name", "unknown"),
            error=reflection.get("error", "Unknown error"),
            reflection=reflection.get("reflection", ""),
            state=failed_state[:800],  # Limit state size
            action=failed_action,
            n_steps=trajectory.get("n_steps", 0),
            invalid_actions=trajectory.get("invalid_actions", 0),
            all_actions=", ".join(trajectory.get("actions", [])[:5]),  # First 5 actions
        )
        
        return prompt
    
    def _parse_claude_suggested_action(self, response: str, fallback_action: str) -> Optional[str]:
        """
        Parse Claude's response to extract the suggested real executable action.
        
        Returns the action string if valid, None otherwise.
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

