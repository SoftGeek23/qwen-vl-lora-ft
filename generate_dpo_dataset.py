#!/usr/bin/env python3
"""
Script to generate DPO dataset from experiment results using reflection-based approach.

This script:
1. Uses ReflectorAgent to analyze failures and generate reflections
2. Uses Claude Sonnet 3.7 thinking to generate high-quality DPO examples
3. Outputs DPO-formatted JSONL files for fine-tuning
"""

import sys
import os
from pathlib import Path

# Use local source code instead of installed package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agisdk.REAL.demo_agent.dpo_dataset_generator import DPODatasetGenerator
from agisdk.REAL.demo_agent.reflector_agent import ReflectorAgent, TaskLog
from agisdk.REAL.demo_agent.memory_index import MemoryIndex


def main():
    """Generate DPO dataset using reflection-based approach."""
    
    # Configuration
    results_dir = "./results"
    output_file = "./dpo_dataset.jsonl"
    min_steps = 1
    reflector_model = "sonnet-3.7:thinking"  # Model for failure reflection
    dpo_model = "sonnet-3.7:thinking"  # Model for DPO generation
    
    # Check for API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("⚠️  ERROR: ANTHROPIC_API_KEY environment variable is not set!")
        print()
        print("Please set your Anthropic API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print()
        print("Or run the script with:")
        print("  ANTHROPIC_API_KEY='your-key' python generate_dpo_dataset.py")
        print()
        return
    
    print("🔍 Generating DPO dataset using reflection-based approach...")
    print(f"   Results directory: {results_dir}")
    print(f"   Output file: {output_file}")
    print(f"   Reflector model: {reflector_model}")
    print(f"   DPO generation model: {dpo_model}")
    print()
    
    # Step 1: Collect task logs
    print("📋 Step 1: Collecting task logs from experiments...")
    from agisdk.REAL.harness import harness
    
    # Create a temporary harness to collect results
    # In practice, you'd load existing results
    results = {}
    
    # Try to load results from directory
    results_path = Path(results_dir)
    if results_path.exists():
        # Find all experiment directories
        for exp_dir in results_path.rglob("summary_info.json"):
            exp_parent = exp_dir.parent
            try:
                import json
                with open(exp_dir, 'r') as f:
                    summary = json.load(f)
                    task_name = summary.get("task_name", "unknown")
                    
                    # Fallback: extract from directory name if not in summary
                    if task_name == "unknown":
                        dir_name = exp_parent.name
                        # Format: 2025-12-21_01-43-53_DemoAgentArgs_on_v1.omnizon-1_937_...
                        if "_on_" in dir_name:
                            parts = dir_name.split("_on_")
                            if len(parts) > 1:
                                task_name = parts[1].split("_")[0]  # e.g., "v1.omnizon-1"
                    
                    results[task_name] = {
                        "cum_reward": summary.get("cum_reward", 0),
                        "n_steps": summary.get("n_steps", 0),
                        "elapsed_time": summary.get("elapsed_time", 0),
                        "exp_dir": str(exp_parent),
                        "err_msg": summary.get("err_msg"),
                    }
                    status = "✓" if results[task_name]["cum_reward"] == 1 else "✗"
                    print(f"   {status} Loaded {task_name}: {results[task_name]['n_steps']} steps")
            except Exception as e:
                print(f"   ⚠️  Failed to load {exp_dir}: {e}")
    
    if not results:
        print("   ⚠️  No results found. Please run experiments first.")
        return
    
    print(f"   Found {len(results)} completed tasks")
    
    # Step 2: Create reflector and analyze failures
    print()
    print("🔬 Step 2: Reflecting on failures...")
    
    # Initialize memory index (not used for DPO, but required by ReflectorAgent)
    memory_index = MemoryIndex(memory_dir="./agent_memories")
    
    # Create reflector agent
    reflector = ReflectorAgent(
        memory_index=memory_index,
        reflector_model=reflector_model,
        use_llm_reflection=True,
        anthropic_api_key=anthropic_api_key,
    )
    
    # Collect task logs
    task_logs = reflector.collect_task_logs(results, results_dir=results_dir)
    
    # Reflect on failures
    failure_reflections = reflector.reflect_on_failures(task_logs)
    
    print(f"   Generated {len(failure_reflections)} failure reflections")
    
    if not failure_reflections:
        print("   ⚠️  No failures to reflect on. Using direct generation method.")
        # Fallback to direct generation
        generator = DPODatasetGenerator(
            results_dir=results_dir,
            output_file=output_file,
            min_steps=min_steps,
        )
        output_path = generator.generate_and_save()
    else:
        # Step 3: Generate DPO dataset from reflections
        print()
        print("🤖 Step 3: Generating DPO examples with Claude Sonnet 3.7 thinking...")
        
        # Collect successful steps for reference
        generator = DPODatasetGenerator(
            results_dir=results_dir,
            output_file=output_file,
            min_steps=min_steps,
            dpo_model=dpo_model,
            anthropic_api_key=anthropic_api_key,
        )
        
        # Get successful steps for pairing
        successful_steps = []
        for exp_dir in generator._find_experiment_dirs():
            summary_info = generator._load_summary_info(exp_dir)
            if summary_info and summary_info.get("cum_reward", 0) == 1:
                steps = generator._load_all_steps(exp_dir)
                for step_idx, step_info in enumerate(steps):
                    state = generator._extract_state_summary(step_info)
                    action = generator._extract_action(step_info)
                    if state and action:
                        successful_steps.append((
                            step_info,
                            {
                                "task_name": summary_info.get("task_name", "unknown"),
                                "step": step_idx,
                                "exp_dir": str(exp_dir),
                                "success": True,
                            }
                        ))
        
        # Generate DPO dataset from reflections
        print(f"   Processing {len(failure_reflections)} failure reflections...")
        
        # Enable verbose logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s - %(name)s - %(message)s'
        )
        logger = logging.getLogger("agisdk.REAL.demo_agent.dpo_dataset_generator")
        logger.setLevel(logging.INFO)  # Use INFO level to see progress
        
        examples = generator.generate_from_reflections(
            failure_reflections=failure_reflections,
            successful_steps=successful_steps,
        )
        
        print(f"   Generated {len(examples)} DPO examples")
        
        if len(examples) == 0:
            print("\n   🔍 Debugging: Checking why no examples were generated...")
            # Check if trajectories can be loaded
            for i, reflection in enumerate(failure_reflections[:3]):  # Check first 3
                exp_dir = reflection.get("exp_dir")
                if exp_dir:
                    print(f"   Checking reflection {i+1}: {reflection.get('task_name', 'unknown')}")
                    print(f"     exp_dir: {exp_dir}")
                    trajectory = generator._extract_trajectory(Path(exp_dir))
                    if trajectory:
                        actions = trajectory.get('actions', [])
                        valid_actions = [a for a in actions if a and not a.startswith("noop(")]
                        print(f"     ✓ Trajectory loaded: {len(actions)} total actions, {len(valid_actions)} valid (non-noop)")
                        print(f"     First 3 actions: {actions[:3]}")
                        if valid_actions:
                            print(f"     First valid action: {valid_actions[0]}")
                            # Test Claude call
                            if generator.claude_client:
                                print(f"     Testing Claude generation...")
                                try:
                                    dpo = generator._generate_dpo_with_claude_reflection(
                                        reflection=reflection,
                                        failed_state=trajectory.get("states", [""])[0] if trajectory.get("states") else "Task: test",
                                        failed_action=valid_actions[0],
                                        trajectory=trajectory,
                                    )
                                    if dpo:
                                        print(f"     ✓ Claude generated example: chosen={dpo.chosen}")
                                    else:
                                        print(f"     ✗ Claude failed to generate example")
                                except Exception as e:
                                    print(f"     ✗ Claude error: {e}")
                            else:
                                print(f"     ✗ Claude client not initialized")
                        else:
                            print(f"     ✗ No valid actions (all are noop)")
                    else:
                        print(f"     ✗ Could not load trajectory")
        
        if examples:
            generator.save_dataset(examples)
            output_path = generator.output_file
        else:
            print("   ⚠️  No examples generated! Check logs for details.")
            output_path = generator.output_file
            # Create empty file to indicate failure
            with open(output_path, 'w') as f:
                pass
    
    print()
    print(f"✅ DPO dataset generated successfully!")
    print(f"   Saved to: {output_path}")
    print()
    print("📊 Dataset format:")
    print("   Each line is a JSON object with:")
    print("   - prompt: The state/context the agent observed")
    print("   - chosen: The preferred action (what should have been done)")
    print("   - rejected: The non-preferred action (what actually failed)")
    print("   - metadata: Additional info (task_name, reasoning, etc.)")
    print()
    print("💡 Usage for fine-tuning:")
    print("   This JSONL file can be used with DPO training libraries like:")
    print("   - TRL (Transformers Reinforcement Learning)")
    print("   - Axolotl")
    print("   - Unsloth")


if __name__ == "__main__":
    main()

