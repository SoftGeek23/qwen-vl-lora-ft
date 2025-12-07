<p align="center">
  <h1 align="center">🚀 AGI SDK</h1>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2504.11543">📄 Paper</a> •
  <a href="https://www.theagi.company/blog/introducing-real-bench">📝 Blog</a> •
  <a href="https://www.theagi.company">🏢 AGI Inc</a> •
  <a href="https://www.realevals.ai">🏆 Leaderboard</a>
</p>


<p align="center">
  <a href="https://pypi.org/project/agisdk"><img src="https://img.shields.io/pypi/v/agisdk?color=brightgreen" alt="PyPI version"></a>
  <a href="https://pypi.org/project/agisdk"><img src="https://img.shields.io/pypi/pyversions/agisdk" alt="Python versions"></a>
  <a href="https://static.pepy.tech/badge/agisdk"><img src="https://static.pepy.tech/badge/agisdk" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/agi-inc/agisdk" alt="License"></a>
</p>

<p align="center">
  <b>Build, evaluate, and level up your AI agents for the real web.</b>
</p>

<p align="center">
  <img src="docs/images/real.gif" alt="REAL benchmark demo" width="600">
</p>




# ✨ What is AGI SDK?

**AGI SDK** is a toolkit for **building** and **evaluating** AI browser agents in real-world environments.

It powers [REAL Bench](https://realevals.xyz): the first high-fidelity benchmark for AI agents navigating modern websites like Amazon, DoorDash, Airbnb, and more.

🔹 **Train agents** to browse and interact with real apps
🔹 **Benchmark agents** with robust, standardized tasks
🔹 **Submit to the leaderboard** and see how your agents stack up!

> **TL;DR**: Go from “idea” to “benchmarked agent” in <60 seconds

## 🛠️ Installation (30 s)

```bash
# Install the SDK
pip install agisdk

# Install Playwright browser dependencies
playwright install --force

# Set your LLM API key (for evaluation)
export OPENAI_API_KEY="your-api-key"   # any supported provider key works
```

✅ Supports OpenAI, Anthropic, OpenRouter, and custom models! <br>

On Apple Silicon run `brew install --cask playwright` first.


## ⏱️ 60-second Quick-Start

Here's a minimal example to get you started for benchmarking an AI agent on the REAL Bench environment:

```python
from agisdk import REAL

harness = REAL.harness(
    model="gpt-4o",       # any LLM tag
    task_type="omnizon",  # Amazon-like store
    headless=False        # watch it click in real-time!
)

print(harness.run())      # 🎉
```
Need more control? [See full examples ›](/example)

## 🔥 Features

- Full-stack **web replicas** of top real-world apps (Amazon, Uber, Gmail, Airbnb, etc.)
- **Robust agent API**: Observations, Actions, Memory, Errors
- **Leaderboard integration** (REAL Bench)
- **Customizable harness**: plug your own agents
- **Multi-model support**: OpenAI, Anthropic, OpenRouter, or your own model
- **Parallel evaluation** for faster experiments



## Running Custom Agents

Checkout the README.md in the `example` folder. There are three examples of custom agents in the `example` directory:

- `example/starter.py`: A simple example to get you started
- `example/custom.py`: A more complex example with a custom agent
- `example/nova.py`: For running custom agents which already have browsers running (in this case, Amazon NovaAct)

Additionally, there is a hackable example in `example/hackable.py` which is a can be configured for better performance and starting of.

## Local Development

Only if you want to develop locally, you can install from source:

```bash
# Clone the repository
git clone https://github.com/agi-inc/agisdk.git
cd agisdk

# Install in development mode
pip install -e .
```

## 🌐 Available Tasks

> **Versioning:** The SDK ships both `v1` and `v2` task sets; if you omit the version when selecting tasks or running experiments the harness defaults to `v1`. Specify `task_version="v2"` (or use `v2.*` task ids) to target the newer scenarios.

The AGI SDK includes high-fidelity, fully-deterministic websites for agents to explore. These are modern web stack sites (React + Next.js) with rich functionality for core user flows, realistic mock data, and consistent behavior for testing and evaluation.

The benchmark includes these environments:

| App Clone | Task Prefix | Example Use Case |
| :--- | :--- | :--- |
| 🛒 Amazon → Omnizon | `v2.omnizon-*` | Buy a laptop, find a gift |
| 🍔 DoorDash → DashDish | `v2.dashdish-*` | Order dinner |
| ✈️ United → FlyUnified | `v2.flyunified-*` | Book a flight |
| 🏡 Airbnb → Staynb | `v2.staynb-*` | Reserve accommodation |
| 📅 Google Calendar → GoCalendar | `v2.gocalendar-*` | Schedule a meeting |
| 📬 Gmail → GoMail | `v2.gomail-*` | Compose an email |
| 🍽️ OpenTable → OpenDining | `v2.opendining-*` | Book a restaurant |
| 👔 LinkedIn → NetworkIn | `v2.networkin-*` | Accept a connection |
| 🚗 Uber → Udriver | `v2.udriver-*` | Book a ride |
| 💼 UpWork → TopWork | `v2.topwork-*` | Find a freelance gig |
| 🏠 Zillow → Zilloft | `v2.zilloft-*` | Browse houses |

Each task comes with **human-written goals** designed to stress-test agent capabilities.

## 🔑 API Keys

To use models from other providers, set their respective API keys:

```bash
# For Anthropic models (like sonnet-3.7)
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 🖥️ Running Local Models with vLLM

You can run local models using vLLM server. This is especially useful for vision-language models like Qwen2.5-VL.

### Setting up vLLM Server

1. **Install vLLM** (with vision support for vision-language models):

```bash
# For vision-language models like Qwen2.5-VL
pip install "vllm[vision]>=0.6.0"

# Or for standard models
pip install vllm>=0.6.0
```

2. **Start the vLLM server**:

For Qwen2.5-VL-32B-Instruct-AWQ (vision-language model):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --port 7999 \
    --trust-remote-code \
    --dtype auto
```

For standard text models:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <model-name> \
    --port 7999 \
    --trust-remote-code \
    --dtype auto
```

**Note:** The default port is `7999`. If you use a different port, you'll need to modify the `base_url` in the agent configuration.

3. **Use the local model in your code**:

```python
from agisdk import REAL

harness = REAL.harness(
    model="local/Qwen/Qwen2.5-VL-32B-Instruct-AWQ",  # Prefix with "local/"
    headless=True,
    max_steps=25,
    use_screenshot=True,
    use_axtree=True,
)

print(harness.run())
```

The SDK will automatically connect to `http://localhost:7999/v1` when you use the `local/` prefix.

### vLLM Server Options

Common vLLM server options you might need:

- `--port`: Port number (default: 8000, SDK expects 7999)
- `--trust-remote-code`: Required for models with custom code
- `--dtype`: Data type (`auto`, `float16`, `bfloat16`)
- `--max-model-len`: Maximum sequence length
- `--enforce-eager`: Use eager mode (useful for debugging)
- `--gpu-memory-utilization`: GPU memory utilization (0.0-1.0)
- `--tensor-parallel-size`: Number of tensor parallel replicas

For AWQ quantized models, vLLM will automatically detect and use the quantization.

### Running with Visible Browser (headless=False)

If you want to see the browser window (`headless=False`), you need a display server. On headless Linux servers, you have several options:

**Option 1: Use xvfb-run (Virtual Display)**
```bash
# Install xvfb if not already installed
sudo apt-get install xvfb

# Run your script with xvfb-run
xvfb-run -a python starter.py
```
Note: This creates a virtual display, so the browser runs but you won't see it visually. Useful for debugging without a GUI.

**Option 2: X11 Forwarding (if SSH'd in)**
```bash
# Connect with X11 forwarding enabled
ssh -X user@server

# Or with trusted X11 forwarding
ssh -Y user@server

# Then run normally
python starter.py
```

**Option 3: Use VNC or Remote Desktop**
Set up a VNC server or remote desktop environment on your server, then connect to view the desktop.

**Option 4: Run on a machine with a display**
If you have a local machine with a GUI, run the script there instead of on a headless server.

**For headless servers without display access:**
Simply use `headless=True` in your harness configuration - the browser will still work, you just won't see the window.

## 👁️ Observation Structure

Your agent gets access to the following observation structure:

```python
{
    'chat_messages': [...],          # History of chat messages
    'goal': "...",                   # Text description of the goal
    'goal_object': [...],            # Structured goal object with text and images
    'open_pages_urls': [...],        # List of open page URLs
    'active_page_index': 0,          # Index of the active page
    'url': "...",                    # Current URL
    'screenshot': np.array(...),     # Screenshot as numpy array
    'dom_object': {...},             # DOM structure
    'axtree_object': {...},          # Accessibility tree
    'extra_element_properties': {...}, # Additional element properties
    'focused_element_bid': "...",    # ID of the focused element
    'last_action': "...",            # Last action performed
    'last_action_error': "...",      # Error from last action (if any)
    'elapsed_time': 0.0,             # Time elapsed in the episode
    'browser': {...}                 # Playwright browser object (for direct control)
}
```

## 🎯 Actions

Actions are specified as strings in the format of function calls. Here are some commonly used actions:

```python
# Navigation
"goto('https://www.google.com')"
"go_back()"
"go_forward()"

# Interaction
"click('element_id')"
"fill('input_id', 'text to enter')"
"press('Enter')"

# Communication
"send_msg_to_user('I found the answer: $42.99')"

# Reporting infeasible tasks
"report_infeasible('The requested item is out of stock')"
```

## ⚙️ Harness Configuration

The harness function accepts the following parameters:

```python
REAL.harness(
    # Agent configuration (provide one of these)
    model="gpt-4o",                                # OpenAI models
    model="sonnet-3.7",                            # Anthropic models
    model="openrouter/deepseek/deepseek-chat-v3-0324", # OpenRouter models (with openrouter/ prefix)
    agentargs=MyAgentArgs(),                       # Or provide your own agent arguments

    # Task selection (provide one of these or don't provide any to run all tasks)
    task_name="v2.omnizon-1",  # Specific task to run
    task_type="omnizon",              # Run all tasks of this type
    task_id=1,                        # Run specific task ID within a type

    # Browser configuration
    headless=False,                   # Whether to show the browser (requires X server or xvfb-run)
    max_steps=25,                     # Maximum number of steps
    browser_dimensions=(1280, 720),   # Browser window dimensions

    # Observation options
    use_html=False,                   # Include HTML in observations
    use_axtree=True,                  # Include accessibility tree
    use_screenshot=True,              # Include screenshots

    # Leaderboard submission
    leaderboard=False,                # Whether to submit to leaderboard
    run_id="my_unique_id",            # Unique ID for the submission

    # Execution options
    num_workers=4,                    # Number of parallel workers
    use_cache=True,                   # Use cached results when available
    cache_only=False,                 # Only use cached results
    force_refresh=False,              # Force re-running tasks

    # Output options
    results_dir="./results"           # Where to store results
)
```

## 🏆 Submitting to the REAL Leaderboard

1. **Create an API key** – use the leaderboard portal (Account → API Keys) to generate a key tied to your Supabase user.
2. **Mint a run ID**
   - **From the portal UI:** open the Profile page, click **Create Run**, pick your model, and copy the `run_id` that appears in the runs table.
   - **From the API (same endpoint the SDK uses):**
   ```bash
   curl "https://www.realevals.ai/api/runKey?api_key=<API_KEY>&model_name=<MODEL_NAME>&run_name=<RUN_NAME>"
   ```
   The JSON response returns `newRunId`. If want to use a different domain, set `REAL_API_BASE=https://…` before running the SDK to override the default domain.
3. **Run the harness in leaderboard mode**:
   ```python
   harness = REAL.harness(
       model="gpt-4o",
       task_type="omnizon",
       leaderboard=True,
       api_key="<API_KEY>",
       run_name="<RUN_NAME>",
       model_id_name="<MODEL_NAME>",
       run_id="<newRunId>",
   )
   harness.run()
   ```
   The harness sets `RUNID` so each clone posts results to the REAL API. Use `force_refresh=True` or delete cached runs in `example/results/` when you need a fresh submission.
4. **Inspect the submission** – either open the leaderboard UI or call
   ```
   https://web-eval-leaderboard.vercel.app/api/getRunTask?api_key=<API_KEY>&display_name=<RUN_NAME>&task_id=<TASK_ID>
   ```
   to fetch stored results (use bare task IDs such as `omnizon-1`; inside the SDK you reference tasks with the `v2.` prefix).


## 🤝 Contributing

We welcome contributions of all kinds:
- 📢 Feature requests? [Open an Issue](https://github.com/agi-inc/agisdk/issues)
- 🐛 Bug reports? [Create a ticket](https://github.com/agi-inc/agisdk/issues)
- 📈 Improve REAL tasks? Join our [Project Board](https://github.com/orgs/agi-inc/projects/2)
- 🛠️ Submit code? Fork + PR - we love clean commits!

Let's build the future of agents together. 🔥

## 💬 Community
- [Join our Discord](https://discord.gg/c95EJDfXzx) (_coming soon!_)
- [Follow AGI Inc. on LinkedIn](https://www.linkedin.com/company/the-agi-company/)

## ⭐️ Why AGI SDK?

Because **your agents deserve better** than toy environments. <br>
Because **the real web is messy** and that's where the magic happens. <br>
Because **the future is agentic** and it starts here.
