"""
Agent with Memory Index Integration

This extends the basic agent to use a memory index for learning from past experiences.
"""

import logging
from typing import Optional

from .basic_agent import DemoAgent, image_to_jpg_base64_url
from .memory_index import MemoryIndex, MemoryExemplar

logger = logging.getLogger(__name__)


class MemoryAgent(DemoAgent):
    """
    Enhanced agent that uses a memory index to learn from past experiences.
    
    The memory index stores state-action-result-reflection exemplars and retrieves
    similar memories to guide decision-making.
    """
    
    def __init__(
        self,
        *args,
        use_memory: bool = True,
        memory_dir: str = "./agent_memories",
        memory_top_k: int = 3,
        **kwargs,
    ):
        """
        Initialize the memory-enabled agent.
        
        Args:
            *args: Arguments passed to DemoAgent
            use_memory: Whether to enable memory indexing
            memory_dir: Directory to store memories
            memory_top_k: Number of memories to retrieve per decision
            **kwargs: Additional arguments passed to DemoAgent
        """
        super().__init__(*args, **kwargs)
        
        self.use_memory = use_memory
        self.memory_index: Optional[MemoryIndex] = None
        
        if self.use_memory:
            self.memory_index = MemoryIndex(
                memory_dir=memory_dir,
                top_k=memory_top_k,
            )
            logger.info(f"Memory index initialized with {len(self.memory_index.memories)} existing memories")
        
        # Track last state for memory storage
        self.last_state_summary: Optional[str] = None
        self.last_action: Optional[str] = None
    
    def _extract_state_summary(self, obs: dict) -> str:
        """
        Extract a textual summary of the current state for memory indexing.
        
        This should capture key aspects of the page state that would help
        identify similar situations in the future.
        """
        summary_parts = []
        
        # URL
        if obs.get("url"):
            summary_parts.append(f"URL: {obs['url']}")
        
        # Key elements from axtree (first 500 chars)
        if obs.get("axtree_txt"):
            axtree_snippet = obs["axtree_txt"][:500]
            summary_parts.append(f"Page elements: {axtree_snippet}")
        
        # Last action error (if any) - important context
        if obs.get("last_action_error"):
            summary_parts.append(f"Last error: {obs['last_action_error'][:200]}")
        
        # Goal context
        if obs.get("goal_object"):
            goal_text = str(obs["goal_object"])
            if len(goal_text) > 200:
                goal_text = goal_text[:200] + "..."
            summary_parts.append(f"Goal: {goal_text}")
        
        return " | ".join(summary_parts)
    
    def _extract_action_context(self, obs: dict) -> str:
        """
        Extract context about what action we're considering.
        This helps retrieve more relevant memories.
        """
        context_parts = []
        
        # What we're trying to do
        if obs.get("last_action_error"):
            context_parts.append(f"Previous action failed: {obs['last_action_error'][:100]}")
        
        # Current page state hints
        if obs.get("axtree_txt"):
            # Look for common UI patterns
            axtree_lower = obs["axtree_txt"].lower()
            if "search" in axtree_lower:
                context_parts.append("searching")
            if "button" in axtree_lower:
                context_parts.append("button interaction")
            if "input" in axtree_lower or "textbox" in axtree_lower:
                context_parts.append("text input")
        
        return " ".join(context_parts)
    
    def get_action(self, obs: dict) -> tuple[str, dict]:
        """
        Get next action, enhanced with memory retrieval.
        Overrides parent to inject memory context into prompts.
        """
        # Extract state summary for memory
        state_summary = self._extract_state_summary(obs)
        action_context = self._extract_action_context(obs)
        
        # Store for later (to save memory after action completes)
        self.last_state_summary = state_summary
        
        # Build system and user messages (similar to parent but with memory injection)
        system_msgs = []
        user_msgs = []
        
        # Build system messages
        if self.chat_mode:
            system_msgs.append({
                "type": "text",
                "text": """\
# Instructions

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, to which the user gives you instructions and to which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
            })
            # Append chat messages
            user_msgs.append({"type": "text", "text": "# Chat Messages\n"})
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append({
                        "type": "text",
                        "text": f"- [{msg['role']}] {msg['message']}\n",
                    })
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
        else:
            assert obs["goal_object"], "The goal is missing."
            system_msgs.append({
                "type": "text",
                "text": """\
# Instructions

Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
            })
            user_msgs.append({"type": "text", "text": "# Goal\n"})
            user_msgs.extend(obs["goal_object"])
        
        # INJECT MEMORY HERE - before other context
        if self.use_memory and self.memory_index:
            task_type = self._get_task_type_from_obs(obs)
            memory_text = self.memory_index.get_memories_for_prompt(
                state_summary,
                action_context,
                task_type,
            )
            if memory_text:
                user_msgs.append({
                    "type": "text",
                    "text": memory_text + "\n",
                })
        
        # Append page AXTree (if asked)
        if self.use_axtree:
            user_msgs.append({
                "type": "text",
                "text": f"# Current page Accessibility Tree\n\n{obs['axtree_txt']}\n\n",
            })
        
        # Append HTML (if asked)
        if self.use_html:
            user_msgs.append({
                "type": "text",
                "text": f"# Current page HTML\n\n{obs.get('pruned_html', '')}\n\n",
            })
        
        # Append screenshot (if asked)
        if self.use_screenshot and obs.get("screenshot") is not None:
            user_msgs.append({
                "type": "text",
                "text": "# Current page Screenshot\n",
            })
            user_msgs.append({
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(obs["screenshot"]),
                    "detail": "auto",
                },
            })
        
        # Append action space description
        user_msgs.append({
            "type": "text",
            "text": f"# Action Space\n\n{self.action_set.describe(with_long_description=False, with_examples=True)}\n",
        })
        
        # Append past actions
        if self.action_history:
            user_msgs.append({"type": "text", "text": "# History of past actions\n"})
            user_msgs.extend([
                {"type": "text", "text": f"{action}\n"}
                for action in self.action_history
            ])
            
            if obs["last_action_error"]:
                rich_logger.error(f"Error: {str(obs['last_action_error'])[:100]}...")
                user_msgs.append({
                    "type": "text",
                    "text": f"# Error message from last action\n\n{obs['last_action_error']}\n\n",
                })
        
        # Ask for next action
        user_msgs.append({
            "type": "text",
            "text": """# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page, and relevant memories from past experiences before deciding on your next action.
""",
        })
        
        # Query model
        action = self.query_model(system_msgs, user_msgs)
        
        # Extract action type for logging
        action_type = action.split("(")[0] if "(" in action else "unknown"
        action_args = action.split("(", 1)[1].rstrip(")") if "(" in action else ""
        
        # Log action
        step_num = len(self.action_history) + 1
        action_summary = f"{action_type}"
        if action_args:
            action_summary += f"({action_args[:50]}{'...' if len(action_args) > 50 else ''})"
        rich_logger.task_step(step_num, action_summary)
        
        self.action_history.append(action)
        self.last_action = action
        
        # Store observation (will trigger memory storage)
        self.update_last_observation(obs)
        
        return action, {}
    
    def _get_task_type_from_obs(self, obs: dict) -> Optional[str]:
        """Extract task type from observation if available."""
        # Try to extract from URL or other observation fields
        url = obs.get("url", "")
        if "omnizon" in url.lower():
            return "omnizon"
        elif "dashdish" in url.lower():
            return "dashdish"
        # Add more task type detection as needed
        return None
    
    def update_last_observation(self, obs):
        """
        Update observation and store memory if action completed.
        """
        super().update_last_observation(obs)
        
        # Store memory after action completes
        if (
            self.use_memory 
            and self.memory_index 
            and self.last_state_summary 
            and self.last_action
        ):
            # Determine result
            result = "success"
            if obs.get("last_action_error"):
                result = f"failure: {obs['last_action_error'][:200]}"
            
            # Generate reflection
            reflection = self._generate_reflection(obs, result)
            
            # Extract task info
            task_type = self._get_task_type_from_obs(obs)
            task_id = obs.get("task_id")
            
            # Add to memory
            self.memory_index.add_memory(
                state_summary=self.last_state_summary,
                action=self.last_action,
                result=result,
                reflection=reflection,
                task_type=task_type,
                task_id=task_id,
            )
            
            logger.debug(f"Stored memory: {self.last_action} -> {result}")
    
    def _generate_reflection(self, obs: dict, result: str) -> str:
        """
        Generate a reflection on why the action succeeded or failed.
        This helps the model learn from experiences.
        """
        if "success" in result.lower():
            reflection = "Action succeeded. This approach worked well in this context."
        else:
            error = obs.get("last_action_error", "")
            if "multi-action" in error:
                reflection = "Failed because multiple actions were attempted at once. Only one action per step is allowed."
            elif "not an <input>" in error or "not editable" in error:
                reflection = "Failed because tried to fill a non-input element. Need to identify the correct input field element."
            elif "not found" in error.lower() or "detached" in error.lower():
                reflection = "Failed because element was not found or page changed. May need to wait or re-identify elements."
            else:
                reflection = f"Action failed: {error[:150]}. Need to adjust approach."
        
        return reflection

