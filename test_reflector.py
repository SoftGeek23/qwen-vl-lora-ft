#!/usr/bin/env python3
"""
Test script to run reflector agent on existing results and generate memories.
"""

import sys
from pathlib import Path

# Use local source code
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agisdk.REAL.demo_agent.reflector_agent import ReflectorAgent
from agisdk.REAL.demo_agent.memory_index import MemoryIndex
from agisdk.REAL.browsergym.experiments import get_exp_result
import os

def load_results_from_experiments(results_dir="./results"):
    """Load results dictionary from existing experiment directories."""
    results_dir = Path(results_dir)
    results = {}
    
    # Find all experiment directories (those containing exp_args.pkl)
    exp_dirs = list(results_dir.glob("*/exp_args.pkl"))
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    for exp_args_path in exp_dirs:
        exp_dir = exp_args_path.parent
        
        try:
            # Load experiment result
            exp_result = get_exp_result(str(exp_dir))
            exp_args = exp_result.exp_args
            
            # Extract task name from exp_args
            task_name = exp_args.task_name if hasattr(exp_args, 'task_name') else None
            if not task_name:
                # Try to extract from directory name
                dir_name = exp_dir.name
                # Format: 2025-12-16_05-26-32_DemoAgentArgs_on_v1.omnizon-1_331_...
                if "_on_" in dir_name:
                    parts = dir_name.split("_on_")
                    if len(parts) > 1:
                        task_name = parts[1].split("_")[0]  # e.g., "v1.omnizon-1"
            
            if not task_name:
                print(f"⚠️  Could not extract task name from {exp_dir}, skipping")
                continue
            
            # Get summary info (lazy loaded)
            try:
                summary = exp_result.summary_info
            except Exception as e:
                print(f"⚠️  Could not load summary for {exp_dir}: {e}")
                continue
            
            # Create result dictionary in format expected by harness
            result = {
                "cum_reward": summary.get("cum_reward", 0) if summary else 0,
                "n_steps": summary.get("n_steps", 0) if summary else 0,
                "elapsed_time": summary.get("elapsed_time", 0) if summary else 0,
                "exp_dir": str(exp_dir),
                "err_msg": summary.get("err_msg") if summary else None,
            }
            
            results[task_name] = result
            print(f"✓ Loaded {task_name}: {'✓' if result['cum_reward'] == 1 else '✗'} ({result['n_steps']} steps)")
            
        except Exception as e:
            print(f"⚠️  Failed to load {exp_dir}: {e}")
            continue
    
    return results


def main():
    """Run reflector agent on existing results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reflector agent on existing results")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    parser.add_argument("--memory-dir", default="./agent_memories", help="Memory directory")
    parser.add_argument("--reflector-model", default="sonnet-3.7:thinking", help="Model for reflection")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing memories first")
    
    args = parser.parse_args()
    
    # Load results from experiment directories
    print("=" * 60)
    print("Loading results from experiment directories...")
    print("=" * 60)
    results = load_results_from_experiments(args.results_dir)
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"\n✓ Loaded {len(results)} task results")
    
    # Initialize memory index
    memory_index = MemoryIndex(
        memory_dir=args.memory_dir,
        embedding_model="BAAI/bge-large-en-v1.5",
        top_k=3,
    )
    
    if args.clear_existing:
        print("\n🗑️  Clearing existing memories...")
        memory_index.clear_memories()
    
    print(f"📚 Memory index has {len(memory_index.memories)} existing memories")
    
    # Create reflector agent
    print(f"\n🤖 Initializing reflector agent with model: {args.reflector_model}")
    reflector = ReflectorAgent(
        memory_index=memory_index,
        reflector_model=args.reflector_model,
        use_llm_reflection=True,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    
    if not reflector.use_llm_reflection:
        print("⚠️  LLM reflection not available, falling back to rule-based")
    
    # Collect task logs
    print("\n" + "=" * 60)
    print("Collecting task logs...")
    print("=" * 60)
    task_logs = reflector.collect_task_logs(results, results_dir=args.results_dir)
    
    print(f"✓ Collected {len(task_logs)} task logs")
    successes = [log for log in task_logs if log.success]
    failures = [log for log in task_logs if not log.success]
    print(f"  - Successes: {len(successes)}")
    print(f"  - Failures: {len(failures)}")
    
    # Generate memories and strategies
    print("\n" + "=" * 60)
    print("Generating memories and strategies with LLM reflection...")
    print("=" * 60)
    from agisdk.REAL.demo_agent.strategy_manager import StrategyManager
    strategy_manager = StrategyManager(strategy_file=str(Path(args.memory_dir) / "strategies.json"))
    
    new_memories, new_strategies = reflector.reflect_on_logs(task_logs)
    
    if new_memories:
        print(f"\n✓ Generated {len(new_memories)} new memories")
        
        # Add to memory index
        reflector.update_memory_index(new_memories)
    
    if new_strategies:
        print(f"\n✓ Generated {len(new_strategies)} new strategies")
        
        # Add to strategy manager
        strategy_manager.add_strategies(new_strategies)
        
        print("\n" + "=" * 60)
        print("Generated Memories:")
        print("=" * 60)
        for i, mem in enumerate(new_memories, 1):
            print(f"\n### Memory {i}")
            print(f"State: {mem.state_summary}")
            print(f"Action: {mem.action}")
            print(f"Result: {mem.result}")
            print(f"Reflection: {mem.reflection}")
            if mem.task_type:
                print(f"Task Type: {mem.task_type}")
    else:
        print("\n⚠️  No new memories generated")
    
    print(f"\n📚 Total memories in index: {len(memory_index.memories)}")
    print(f"💾 Memories saved to: {Path(args.memory_dir) / 'memories.jsonl'}")


if __name__ == "__main__":
    main()

