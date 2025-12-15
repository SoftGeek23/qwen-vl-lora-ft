"""
Memory Index System for Agent Learning

This module implements a memory index that stores and retrieves state-action-result-reflection exemplars
to guide the agent's decision-making during inference.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using simple text matching for memory retrieval.")


@dataclass
class MemoryExemplar:
    """A single memory exemplar containing state, action, result, and reflection."""
    state_summary: str  # Textual summary of the page state (axtree snippet, URL, etc.)
    action: str  # The action taken
    result: str  # "success" or "failure" or specific error message
    reflection: str  # Why it worked/didn't work, what was learned
    task_type: Optional[str] = None  # e.g., "omnizon", "dashdish", etc.
    task_id: Optional[str] = None  # Specific task identifier
    timestamp: Optional[float] = None  # When this memory was created
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryExemplar':
        return cls(**data)
    
    def to_text(self) -> str:
        """Convert memory to a text format for LLM consumption."""
        return f"""State: {self.state_summary}
Action: {self.action}
Result: {self.result}
Reflection: {self.reflection}"""


class MemoryIndex:
    """
    Memory index that stores and retrieves exemplars based on state similarity.
    
    Uses semantic embeddings to find similar states and retrieve relevant memories.
    """
    
    def __init__(
        self,
        memory_dir: str = "./agent_memories",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_embeddings: bool = True,
        top_k: int = 3,
    ):
        """
        Initialize the memory index.
        
        Args:
            memory_dir: Directory to store memory files
            embedding_model: Name of sentence transformer model for embeddings
            use_embeddings: Whether to use semantic embeddings (requires sentence-transformers)
            top_k: Number of top similar memories to retrieve
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "memories.jsonl"
        self.top_k = top_k
        
        self.memories: List[MemoryExemplar] = []
        self.use_embeddings = use_embeddings and HAS_SENTENCE_TRANSFORMERS
        
        if self.use_embeddings:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            self.embeddings: Optional[np.ndarray] = None
        else:
            self.embedder = None
            self.embeddings = None
        
        # Load existing memories
        self.load_memories()
    
    def load_memories(self):
        """Load memories from disk."""
        if not self.memory_file.exists():
            return
        
        self.memories = []
        with open(self.memory_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        self.memories.append(MemoryExemplar.from_dict(data))
                    except json.JSONDecodeError:
                        continue
        
        # Recompute embeddings if using them
        if self.use_embeddings and self.memories:
            self._update_embeddings()
        
        print(f"Loaded {len(self.memories)} memories from {self.memory_file}")
    
    def _update_embeddings(self):
        """Update embeddings for all memories with enhanced context including failures/states."""
        if not self.use_embeddings or not self.embedder:
            return
        
        # Build rich context: state + action + result + reflection
        state_texts = []
        for mem in self.memories:
            parts = [mem.state_summary, mem.action]
            
            # Include failure/state description
            if "failure" in mem.result.lower():
                parts.append(f"Failure: {mem.result}")
            elif "success" in mem.result.lower():
                parts.append("Success")
            else:
                parts.append(f"Result: {mem.result}")
            
            # Include reflection (describes what the state/failure is)
            if mem.reflection:
                parts.append(f"Context: {mem.reflection}")
            
            embedding_text = " | ".join(parts)
            state_texts.append(embedding_text)
        
        self.embeddings = self.embedder.encode(state_texts, convert_to_numpy=True)
    
    def add_memory(
        self,
        state_summary: str,
        action: str,
        result: str,
        reflection: str,
        task_type: Optional[str] = None,
        task_id: Optional[str] = None,
    ):
        """
        Add a new memory exemplar.
        
        Args:
            state_summary: Textual summary of the page state
            action: The action taken
            result: "success" or "failure" or error message
            reflection: Why it worked/didn't work
            task_type: Type of task (e.g., "omnizon")
            task_id: Specific task identifier
        """
        import time
        memory = MemoryExemplar(
            state_summary=state_summary,
            action=action,
            result=result,
            reflection=reflection,
            task_type=task_type,
            task_id=task_id,
            timestamp=time.time(),
        )
        
        self.memories.append(memory)
        
        # Append to file
        with open(self.memory_file, 'a') as f:
            f.write(json.dumps(memory.to_dict()) + '\n')
        
        # Update embeddings if using them
        if self.use_embeddings and self.embedder:
            if self.embeddings is None:
                self._update_embeddings()
            else:
                # Incrementally add embedding with enhanced context
                parts = [state_summary, action]
                if "failure" in result.lower():
                    parts.append(f"Failure: {result}")
                elif "success" in result.lower():
                    parts.append("Success")
                else:
                    parts.append(f"Result: {result}")
                if reflection:
                    parts.append(f"Context: {reflection}")
                
                embedding_text = " | ".join(parts)
                new_embedding = self.embedder.encode(
                    [embedding_text], 
                    convert_to_numpy=True
                )
                self.embeddings = np.vstack([self.embeddings, new_embedding])
    
    def retrieve_similar(
        self,
        current_state_summary: str,
        current_action_context: Optional[str] = None,
        current_error: Optional[str] = None,
        task_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryExemplar]:
        """
        Retrieve similar memories based on current state.
        
        Args:
            current_state_summary: Summary of current page state
            current_action_context: Context about what action we're considering
            task_type: Filter by task type if provided
            top_k: Number of memories to retrieve (overrides default)
        
        Returns:
            List of similar memory exemplars
        """
        if not self.memories:
            return []
        
        top_k = top_k or self.top_k
        
        # Filter by task type if specified
        candidate_memories = self.memories
        if task_type:
            candidate_memories = [m for m in self.memories if m.task_type == task_type]
        
        if not candidate_memories:
            return []
        
        if self.use_embeddings and self.embedder:
            # Build query with error context if present
            query_parts = [current_state_summary]
            if current_action_context:
                query_parts.append(current_action_context)
            if current_error:
                query_parts.append(f"Error: {current_error}")
            
            query_text = " | ".join(query_parts)
            query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)
            
            # Get indices of candidate memories
            candidate_indices = [
                i for i, m in enumerate(self.memories) 
                if m in candidate_memories
            ]
            candidate_embeddings = self.embeddings[candidate_indices]
            
            # Compute cosine similarity
            similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
            similarities = similarities / (
                np.linalg.norm(candidate_embeddings, axis=1) * 
                np.linalg.norm(query_embedding)
            )
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [candidate_memories[i] for i in top_indices]
        else:
            # Simple text-based matching (fallback)
            # Score by keyword overlap
            query_words = set(current_state_summary.lower().split())
            scored = []
            
            for mem in candidate_memories:
                state_words = set(mem.state_summary.lower().split())
                overlap = len(query_words & state_words)
                scored.append((overlap, mem))
            
            scored.sort(reverse=True)
            return [mem for _, mem in scored[:top_k]]
    
    def get_memories_for_prompt(
        self,
        current_state_summary: str,
        current_action_context: Optional[str] = None,
        current_error: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> str:
        """
        Get formatted memories for inclusion in LLM prompt.
        
        Returns a formatted string with relevant memories.
        """
        memories = self.retrieve_similar(
            current_state_summary,
            current_action_context,
            current_error,
            task_type,
        )
        
        if not memories:
            return ""
        
        formatted = "## Relevant Memories from Past Experiences\n\n"
        for i, mem in enumerate(memories, 1):
            formatted += f"### Memory {i}\n{mem.to_text()}\n\n"
        
        formatted += "Use these memories to guide your action. Learn from past successes and failures.\n"
        return formatted
    
    def clear_memories(self):
        """Clear all memories (useful for testing)."""
        self.memories = []
        self.embeddings = None
        if self.memory_file.exists():
            self.memory_file.unlink()
    
    def query_memory(
        self,
        query_text: str,
        task_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Query memory with a natural language description.
        
        Args:
            query_text: Natural language description of what to search for
            task_type: Filter by task type if provided
            top_k: Number of memories to retrieve
            
        Returns:
            Formatted string with relevant memories
        """
        # Use query_text as state summary, no action context
        memories = self.retrieve_similar(
            current_state_summary=query_text,
            current_action_context=None,
            current_error=None,
            task_type=task_type,
            top_k=top_k,
        )
        
        if not memories:
            return ""
        
        formatted = "## Memory Query Results\n\n"
        for i, mem in enumerate(memories, 1):
            formatted += f"### Memory {i}\n{mem.to_text()}\n\n"
        
        formatted += "Use these memories to guide your action. Learn from past successes and failures.\n"
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        if not self.memories:
            return {"total": 0}
        
        stats = {
            "total": len(self.memories),
            "by_result": {},
            "by_task_type": {},
        }
        
        for mem in self.memories:
            stats["by_result"][mem.result] = stats["by_result"].get(mem.result, 0) + 1
            if mem.task_type:
                stats["by_task_type"][mem.task_type] = stats["by_task_type"].get(mem.task_type, 0) + 1
        
        return stats

