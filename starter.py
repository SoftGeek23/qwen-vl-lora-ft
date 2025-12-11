import sys
from pathlib import Path

# Use local source code instead of installed package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agisdk import REAL


def run_agent(api_key=None, run_name=None):
    harness = REAL.harness(
        model="local/google/gemma-3-27b-it",
        task_name="v1.omnizon-2",  # Run a specific omnizon task (change the number for different tasks)
        headless=True,  # Set to True if running on headless server without X server, or use xvfb-run
        max_steps=20,
        use_screenshot=True,
        use_axtree=True,
        num_workers=1,
    )
    return harness.run()


if __name__ == "__main__":
    run_agent()
