import sys
from pathlib import Path

# Use local source code instead of installed package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agisdk import REAL


def run_agent(api_key=None, run_name=None):
    harness = REAL.harness(
        model="local/google/gemma-3-27b-it",  # Agent model (for task execution)
        task_name="v1.omnizon-1",  # Run a specific task (use task_type="omnizon" for all omnizon tasks)
        headless=True,  # Set to True if running on headless server without X server, or use xvfb-run
        max_steps=20,
        use_screenshot=True,
        use_axtree=True,
        num_workers=1,  # Use 1 worker for single task
        use_memory=True,  # Enable memory system
        memory_dir="./agent_memories",
        # Use Claude Sonnet 3.7 thinking for memory reflection
        # Set ANTHROPIC_API_KEY environment variable: export ANTHROPIC_API_KEY="your-key"
        reflector_model="sonnet-3.7:thinking",  # Model for LLM-based reflection
    )
    return harness.run()


if __name__ == "__main__":
    run_agent()
