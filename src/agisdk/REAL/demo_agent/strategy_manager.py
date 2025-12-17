"""
Strategy Manager Module

Manages high-level strategies extracted from task logs.
These strategies are injected into the agent's prompt (not stored as memories).
"""

import json
import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages high-level strategies for agent prompt injection."""
    
    def __init__(self, strategy_file: str = "./agent_memories/strategies.json"):
        """
        Initialize strategy manager.
        
        Args:
            strategy_file: Path to JSON file storing strategies
        """
        self.strategy_file = Path(strategy_file)
        self.strategies: List[str] = []
        self._load_strategies()
    
    def _load_strategies(self):
        """Load strategies from file and consolidate them."""
        if self.strategy_file.exists():
            try:
                with open(self.strategy_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "strategies" in data:
                        self.strategies = data["strategies"]
                    elif isinstance(data, list):
                        self.strategies = data
                    else:
                        self.strategies = []
                
                # Consolidate on load to fix any existing issues
                if self.strategies:
                    self.consolidate_and_reorder()
                
                logger.info(f"Loaded {len(self.strategies)} strategies from {self.strategy_file}")
            except Exception as e:
                logger.warning(f"Failed to load strategies: {e}")
                self.strategies = []
        else:
            self.strategies = []
    
    def add_strategies(self, new_strategies: List[str], validate_universal: bool = True):
        """
        Add new strategies, avoiding duplicates and validating universality.
        
        Args:
            new_strategies: List of strategy strings to add
            validate_universal: If True, reject contextual strategies (default: True)
        """
        # Clean new strategies first (strip prefixes)
        cleaned_new = []
        rejected_contextual = []
        
        for strategy in new_strategies:
            cleaned = self._clean_strategy_text(strategy)
            if cleaned:
                # Validate universality if requested
                if validate_universal and not self._is_universal_strategy(cleaned):
                    rejected_contextual.append(cleaned)
                    logger.warning(
                        f"Rejected contextual strategy (should be a memory instead): {cleaned[:100]}..."
                    )
                    continue
                cleaned_new.append(cleaned)
        
        if rejected_contextual:
            logger.info(
                f"Rejected {len(rejected_contextual)} contextual strategies. "
                "These should be converted to memories instead."
            )
        
        # Use cleaned versions for comparison
        existing_set = set(self._clean_strategy_text(s) for s in self.strategies)
        added_count = 0
        
        for strategy in cleaned_new:
            cleaned = self._clean_strategy_text(strategy)
            if cleaned and cleaned not in existing_set:
                self.strategies.append(cleaned)
                existing_set.add(cleaned)
                added_count += 1
        
        if added_count > 0:
            # Consolidate and reorder after adding
            self.consolidate_and_reorder()
            logger.info(f"Added {added_count} new universal strategies (total: {len(self.strategies)})")
        
        return {
            "added": added_count,
            "rejected": len(rejected_contextual),
            "rejected_strategies": rejected_contextual,
        }
    
    def _clean_strategy_text(self, strategy: str) -> str:
        """
        Clean strategy text by removing "Strategy X:" prefixes.
        
        Args:
            strategy: Raw strategy string
            
        Returns:
            Cleaned strategy string
        """
        if not strategy:
            return ""
        
        strategy = strategy.strip()
        
        # Remove "Strategy X:" prefix if present
        if strategy.startswith("Strategy ") and ":" in strategy:
            parts = strategy.split(":", 1)
            if len(parts) == 2 and parts[0].strip().startswith("Strategy "):
                strategy = parts[1].strip()
        
        return strategy
    
    def _is_universal_strategy(self, strategy: str) -> bool:
        """
        Check if strategy is truly universal (non-contextual).
        
        A strategy is universal if:
        - It applies to ALL task types (not specific to e-commerce, forms, etc.)
        - It contains NO contextual references
        - It's a fundamental behavioral rule
        
        Args:
            strategy: Strategy string to validate
            
        Returns:
            True if strategy is universal, False if it's contextual
        """
        if not strategy:
            return False
        
        strategy_lower = strategy.lower()
        
        # Contextual keywords that indicate the strategy is NOT universal
        contextual_keywords = [
            "when working with",
            "when encountering",
            "for [specific]",
            "on e-commerce",
            "for product",
            "for dropdown",
            "when on page",
            "like [specific]",
            "such as [specific]",
            "for [task type]",
            "when asked to",
            "after performing",
            "when reporting",
            "for search",
            "on [site]",
            "when navigating",
            "pay attention to",
            "for multi-step",
            "when a task takes",
            "if a specific",
            "before interacting",
            "after multiple",
            "when interacting",
            "break down complex",
            "for search interactions",
            "when interacting with elements",
        ]
        
        # Check for contextual phrases
        for keyword in contextual_keywords:
            if keyword in strategy_lower:
                return False
        
        # Reject if contains specific patterns that indicate context
        contextual_patterns = [
            "→",  # Arrow indicating sequence
            "sequence:",
            "pattern:",
            "follow this",
            "follow the",
            "use the pattern",
            "follow a clear sequence",
            "follow sequence",
            "follow strict pattern",
        ]
        
        for pattern in contextual_patterns:
            if pattern in strategy_lower:
                return False
        
        # Reject if mentions specific element types in a contextual way
        if any(phrase in strategy_lower for phrase in [
            "dropdown",
            "select element",
            "search box",
            "search field",
            "form field",
            "button",
            "link",
            "element id",
            "element with",
        ]) and any(phrase in strategy_lower for phrase in [
            "when",
            "for",
            "on",
            "if",
        ]):
            return False
        
        return True
    
    def _save_strategies(self):
        """Save strategies to file."""
        try:
            self.strategy_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.strategy_file, 'w') as f:
                json.dump({"strategies": self.strategies}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save strategies: {e}")
    
    def consolidate_and_reorder(self, remove_contextual: bool = True):
        """
        Consolidate, deduplicate, and reorder strategies.
        
        This method:
        1. Removes "Strategy X:" prefixes from all stored strategies
        2. Filters out contextual strategies (if remove_contextual=True)
        3. Deduplicates semantically similar strategies
        4. Prioritizes critical strategies (e.g., "one action per turn")
        5. Orders strategies logically
        
        Args:
            remove_contextual: If True, remove contextual strategies (default: True)
        """
        if not self.strategies:
            return
        
        # Step 1: Clean all strategies (remove prefixes)
        cleaned = [self._clean_strategy_text(s) for s in self.strategies]
        cleaned = [s for s in cleaned if s]  # Remove empty
        
        # Step 2: Filter out contextual strategies if requested
        if remove_contextual:
            original_count = len(cleaned)
            cleaned = [s for s in cleaned if self._is_universal_strategy(s)]
            removed_count = original_count - len(cleaned)
            if removed_count > 0:
                logger.warning(
                    f"Removed {removed_count} contextual strategies during consolidation. "
                    "These should be converted to memories instead."
                )
        
        # Step 3: Deduplicate exact matches
        seen = set()
        unique = []
        for strategy in cleaned:
            strategy_lower = strategy.lower().strip()
            if strategy_lower not in seen:
                seen.add(strategy_lower)
                unique.append(strategy)
        
        # Step 4: Prioritize critical strategies
        critical_keywords = [
            "one action per turn",
            "single action",
            "multi-action",
            "only one action",
            "execute only one",
        ]
        
        prioritized = []
        remaining = []
        
        for strategy in unique:
            strategy_lower = strategy.lower()
            is_critical = any(keyword in strategy_lower for keyword in critical_keywords)
            if is_critical:
                prioritized.append(strategy)
            else:
                remaining.append(strategy)
        
        # Step 5: Group similar strategies and keep the most concise version
        # Simple deduplication: if one strategy is a substring of another, keep the shorter one
        final = []
        for strategy in prioritized + remaining:
            # Check if this strategy is a substring of any already added
            is_duplicate = False
            for existing in final:
                existing_lower = existing.lower()
                strategy_lower = strategy.lower()
                # If one contains the other, keep the shorter/more specific one
                if strategy_lower in existing_lower and len(strategy) < len(existing):
                    final.remove(existing)
                    final.append(strategy)
                    is_duplicate = True
                    break
                elif existing_lower in strategy_lower:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final.append(strategy)
        
        # Update strategies
        self.strategies = final
        self._save_strategies()
        logger.info(f"Consolidated strategies: {len(final)} unique universal strategies")
    
    def get_strategies_for_prompt(self) -> str:
        """
        Get formatted strategies for inclusion in agent prompt.
        
        Returns:
            Formatted string with strategies, or empty string if none
        """
        if not self.strategies:
            return ""
        
        formatted = "## High-Level Strategies and Best Practices\n\n"
        for i, strategy in enumerate(self.strategies, 1):
            # Strategies are already cleaned, but double-check
            strategy_text = self._clean_strategy_text(strategy)
            formatted += f"{i}. {strategy_text}\n\n"
        
        formatted += "Apply these strategies throughout your task execution.\n"
        return formatted
    
    def clear_strategies(self):
        """Clear all strategies (useful for testing)."""
        self.strategies = []
        if self.strategy_file.exists():
            self.strategy_file.unlink()

