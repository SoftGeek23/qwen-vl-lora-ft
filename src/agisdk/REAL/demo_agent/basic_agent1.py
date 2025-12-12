import base64
import dataclasses
import io
import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional
# 5/10 with, 1/10 without on omnizon

import numpy as np
from PIL import Image

from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
from agisdk.REAL.browsergym.experiments import AbstractAgentArgs, Agent
from agisdk.REAL.browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    prune_html,
)

from ..logging import logger as rich_logger

logger = logging.getLogger(__name__)


def load_memory_exemplars() -> str:
    """
    Load memory exemplars from JSON file and format them for the prompt.
    Returns empty string if file doesn't exist or can't be loaded.
    """
    try:
        # Path relative to this file
        exemplars_path = Path(__file__).parent / "memory_exemplars.json"
        if not exemplars_path.exists():
            return ""
        
        with open(exemplars_path, 'r') as f:
            exemplars = json.load(f)
        
        if not exemplars:
            return ""
        
        formatted = "## Relevant Examples from Past Experiences\n\n"
        formatted += "Learn from these examples to avoid common mistakes:\n\n"
        
        for i, ex in enumerate(exemplars, 1):
            formatted += f"### Example {i}\n"
            formatted += f"**State:** {ex['state']['summary']}\n"
            formatted += f"- URL: {ex['state']['url']}\n"
            formatted += f"- Key elements: {ex['state']['key_elements']}\n"
            formatted += f"**Action:** `{ex['action']}`\n"
            formatted += f"**Result:** {ex['result']}\n"
            formatted += f"**Reflection:** {ex['reflection']}\n\n"
        
        formatted += "Use these examples to guide your actions. Pay special attention to:\n"
        formatted += "- Distinguishing between buttons (for clicking) and input fields (for typing)\n"
        formatted += "- Only executing ONE action per step (no multiple actions together)\n"
        formatted += "- Identifying the correct element type before using fill() or click()\n\n"
        
        return formatted
    except Exception as e:
        logger.warning(f"Failed to load memory exemplars: {e}")
        return ""


# Handling Screenshots
def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class DemoAgent(Agent):
    """A basic agent using OpenAI API, to demonstrate BrowserGym's functionalities."""

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def close(self):
        """Called when the agent is being closed"""
        # Evaluate success if available
        if hasattr(self, "last_observation") and self.last_observation:
            success = self.last_observation.get("success", None)
            reward = self.last_observation.get("reward", 0)
            time_taken = None
            if hasattr(self, "session_start_time"):
                time_taken = time.time() - self.session_start_time

            if success is not None:
                rich_logger.task_complete(success, reward, time_taken)
            else:
                rich_logger.info(f"🎯 Session completed - {len(self.action_history)} actions taken")
        else:
            rich_logger.info(f"🎯 Session completed - {len(self.action_history)} actions taken")

        super().close()

    def update_last_observation(self, obs):
        """Store the last observation for metrics"""
        self.last_observation = obs

    def __init__(
        self,
        model_name: str,
        chat_mode: bool,
        demo_mode: str,
        use_html: bool,
        use_axtree: bool,
        use_screenshot: bool,
        system_message_handling: Literal["separate", "combined"] = "separate",
        thinking_budget: int = 10000,
        openai_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_site_url: Optional[str] = None,
        openrouter_site_name: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.system_message_handling = system_message_handling
        self.thinking_budget = thinking_budget

        if not (use_html or use_axtree):
            raise ValueError("Either use_html or use_axtree must be set to True.")

        import os

        from anthropic import Anthropic
        from openai import OpenAI

        if (
            model_name.startswith("gpt-")
            or model_name.startswith("o1")
            or model_name.startswith("o3")
        ):
            # Use provided API key or fall back to environment variable
            self.client = OpenAI(api_key=openai_api_key)
            self.model_name = model_name

            # Define function to query OpenAI models
            def query_model(system_msgs, user_msgs):
                if self.system_message_handling == "combined":
                    # Combine system and user messages into a single user message
                    combined_content = ""
                    if system_msgs:
                        combined_content += system_msgs[0]["text"] + "\n\n"
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            combined_content += msg["text"] + "\n"
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": combined_content},
                        ],
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_msgs},
                            {"role": "user", "content": user_msgs},
                        ],
                    )
                return response.choices[0].message.content

            self.query_model = query_model

        elif model_name.startswith("openrouter/"):
            # Extract the actual model name without the openrouter/ prefix
            actual_model_name = model_name.replace("openrouter/", "", 1)

            # Initialize OpenRouter client (using OpenAI client with custom base URL)
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
            )
            # Store site info for headers
            self.openrouter_site_url = openrouter_site_url or os.getenv("OPENROUTER_SITE_URL", "")
            self.openrouter_site_name = openrouter_site_name or os.getenv(
                "OPENROUTER_SITE_NAME", ""
            )
            self.model_name = actual_model_name

            # Define function to query OpenRouter models
            def query_model(system_msgs, user_msgs):
                if self.system_message_handling == "combined":
                    # Combine system and user messages into a single user message
                    combined_content = ""
                    if system_msgs:
                        combined_content += system_msgs[0]["text"] + "\n\n"
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            combined_content += msg["text"] + "\n"
                    response = self.client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": self.openrouter_site_url,
                            "X-Title": self.openrouter_site_name,
                        },
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": combined_content},
                        ],
                    )
                else:
                    # Format messages properly - extract text content
                    formatted_messages = []
                    if system_msgs:
                        formatted_messages.append(
                            {"role": "system", "content": system_msgs[0]["text"]}
                        )

                    # Convert user messages to OpenAI format
                    user_content = []
                    for msg in user_msgs:
                        if msg["type"] == "text":
                            user_content.append({"type": "text", "text": msg["text"]})
                        elif msg["type"] == "image_url":
                            user_content.append(
                                {"type": "image_url", "image_url": msg["image_url"]}
                            )

                    formatted_messages.append({"role": "user", "content": user_content})

                    response = self.client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": self.openrouter_site_url,
                            "X-Title": self.openrouter_site_name,
                        },
                        model=self.model_name,
                        messages=formatted_messages,
                    )
                return response.choices[0].message.content

            self.query_model = query_model

        elif model_name.startswith("local"):
            actual_model_name = model_name.replace("local/", "", 1)

            # Modify OpenAI's API key and API base to use vLLM's load balancer server.
            self.client = OpenAI(
                base_url="http://localhost:7999/v1",
                api_key="FEEL_THE_AGI",
            )
            self.model_name = actual_model_name

            # Define function to query OpenRouter models
            def query_model(system_msgs, user_msgs):
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_msgs},
                        {"role": "user", "content": user_msgs},
                    ],
                    max_tokens=500,
                    temperature=1.0,
                    top_p=0.95,
                    extra_body={"top_k": 64},
                )
                return completion.choices[0].message.content

            self.query_model = query_model

        elif any(model_name.startswith(prefix) for prefix in ["claude-", "sonnet-"]):
            # Comprehensive model mapping for all Claude models
            ANTHROPIC_MODELS = {
                "claude-3-opus": "claude-3-opus-20240229",
                "claude-3-sonnet": "claude-3-sonnet-20240229",
                "claude-3-haiku": "claude-3-haiku-20240307",
                "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
                "claude-opus-4": "claude-opus-4-20250514",
                "claude-sonnet-4": "claude-sonnet-4-20250514",
                "sonnet-3.7": "claude-3-7-sonnet-20250219",
            }

            # Parse model name and thinking mode
            base_model_name = model_name.replace(":thinking", "")
            thinking_enabled = model_name.endswith(":thinking")

            # Get the actual model ID
            if base_model_name in ANTHROPIC_MODELS:
                self.model_name = ANTHROPIC_MODELS[base_model_name]
            else:
                # If not in mapping, assume it's a direct model ID
                self.model_name = base_model_name

            # Initialize Anthropic client
            self.client = Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))

            # Configure thinking based on model capabilities and user request
            if thinking_enabled:
                thinking = {"type": "enabled", "budget_tokens": self.thinking_budget}
            else:
                thinking = {"type": "disabled"}

            # Define function to query Anthropic models
            def query_model(system_msgs, user_msgs):
                # Convert OpenAI format messages to Anthropic format
                anthropic_content = []
                for msg in user_msgs:
                    if msg["type"] == "text":
                        anthropic_content.append({"type": "text", "text": msg["text"]})
                    elif msg["type"] == "image_url":
                        # Handle base64 image URLs for Anthropic
                        image_url = msg["image_url"]
                        if isinstance(image_url, dict):
                            image_url = image_url["url"]

                        if image_url.startswith("data:image/jpeg;base64,"):
                            base64_data = image_url.replace("data:image/jpeg;base64,", "")
                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_data,
                                    },
                                }
                            )
                        else:
                            # Skip external URLs or unsupported image formats
                            anthropic_content.append(
                                {
                                    "type": "text",
                                    "text": "[Image URL not supported by Anthropic API]",
                                }
                            )

                # Handle system message based on system_message_handling
                if self.system_message_handling == "combined" and system_msgs:
                    # Prepend system message to user content
                    combined_content = [{"type": "text", "text": system_msgs[0]["text"]}]
                    combined_content.extend(anthropic_content)
                    anthropic_content = combined_content
                    system_content = None
                else:
                    # Use separate system message - extract text from the list
                    system_content = system_msgs[0]["text"] if system_msgs else ""

                # Simple messages array - no manual thinking block management
                messages = [{"role": "user", "content": anthropic_content}]

                # Make API request - conditionally include system parameter
                create_params = {
                    "model": self.model_name,
                    "max_tokens": 8000,
                    "messages": messages,
                    "thinking": thinking,
                }

                # Only add system parameter if we have content
                if system_content is not None:
                    create_params["system"] = system_content

                response = self.client.messages.create(**create_params)

                # Log response content types for debugging
                logger.info(
                    f"Response content types: {[content.type for content in response.content]}"
                )

                # Extract text content, handling all block types properly
                text_content = None
                for content_block in response.content:
                    if content_block.type == "text":
                        text_content = content_block.text
                        break
                    elif content_block.type == "thinking":
                        # Log thinking content for debugging but don't return it
                        logger.debug(f"Thinking block: {content_block.thinking[:100]}...")
                    elif content_block.type == "redacted_thinking":
                        # Log that we encountered redacted thinking
                        logger.debug("Encountered redacted thinking block")

                if text_content is None:
                    logger.warning("No text content found in response")
                    # This shouldn't happen with a properly formed response
                    raise ValueError("No text content in Anthropic response")

                return text_content

            self.query_model = query_model
        else:
            raise ValueError(
                f"Model {model_name} not supported. Use a model name starting with 'gpt-', 'claude-', 'sonnet-', or 'openrouter/' followed by the OpenRouter model ID."
            )

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas"],  # define a subset of the action space
            # subsets=["chat", "bid", "coord", "infeas"] # allow the agent to also use x,y coordinates
            strict=False,  # less strict on the parsing of the actions
            multiaction=False,  # does not enable the agent to take multiple actions at once
            demo_mode=demo_mode,  # add visual effects
        )
        # use this instead to allow the agent to directly use Python code
        # self.action_set = PythonActionSet())

        self.action_history = []
        self.last_observation = None

    def get_action(self, obs: dict) -> tuple[str, dict]:
        # Print task start information if this is the first action
        if len(self.action_history) == 0:
            goal_str = str(obs.get("goal_object", ""))
            # Extract just the text content if it's a message object
            if isinstance(obs.get("goal_object"), list) and len(obs.get("goal_object")) > 0:
                if (
                    isinstance(obs.get("goal_object")[0], dict)
                    and "text" in obs.get("goal_object")[0]
                ):
                    goal_str = obs.get("goal_object")[0]["text"]
            rich_logger.task_start(goal_str, self.model_name)
            self.session_start_time = time.time()

        system_msgs = []
        user_msgs = []

        if self.chat_mode:
            system_msgs.append(
                {
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
                }
            )
            # append chat messages
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
                            # Chat Messages
                            """,
                }
            )
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
                                    - [{msg["role"]}] {msg["message"]}
                                    """,
                        }
                    )
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

        else:
            assert obs["goal_object"], "The goal is missing."
            system_msgs.append(
                {
                    "type": "text",
                    "text": """\
                            # Instructions

                            Review the current state of the page and all other information to find the best
                            possible next action to accomplish your goal. Your answer will be interpreted
                            and executed by a program, make sure to follow the formatting instructions.
                            
                            CRITICAL: If your goal involves retrieving, listing, or finding information (like product listings, search results, or data), you must:
                            1. Navigate to the correct page/location
                            2. Actually read and extract the information from the page (using the accessibility tree, HTML, or screenshot)
                            3. Include the actual information (names, details, items) in your final send_msg_to_user() message
                            
                            Do NOT just navigate and say you're ready - you must actually extract and communicate the information.
                            """,
                }
            )
            # append goal
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
                            # Goal
                            """,
                }
            )
            # goal_object is directly presented as a list of openai-style messages
            user_msgs.extend(obs["goal_object"])

        # append page AXTree (if asked)
        if self.use_axtree:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # Current page Accessibility Tree

                            {obs["axtree_txt"]}

                            NOTE: To find input fields, search this tree text for the word "textbox". Elements are shown as "[bid] role 'name' (description)". When you see a line with "textbox", use that element's bid with fill().
                            """,
                }
            )
        # append page HTML (if asked)
        if self.use_html:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
                            # Current page DOM

                            {obs["pruned_html"]}

                            """,
                }
            )

        # append page screenshot (if asked)
        if self.use_screenshot:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
                            # Current page Screenshot
                            """,
                }
            )
            user_msgs.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot"]),
                        "detail": "auto",
                    },  # Literal["low", "high", "auto"] = "auto"
                }
            )

        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
                        # Action Space

                        {self.action_set.describe(with_long_description=False, with_examples=True)}

                        Here are examples of actions with chain-of-thought reasoning:

                        I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
                        ```click("12")```

                        I found the information requested by the user, I will send it to the chat.
                        ```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

                        IMPORTANT: When searching, if you click a search button and a search input field appears, you MUST find the textbox element (role='textbox') in the updated accessibility tree and use fill() on it. Example:
                        Step 1: Click the search button to open the search interface.
                        ```click("166")```
                        Step 2: After clicking, look for the textbox element in the new accessibility tree (it will have role='textbox'), then fill it with your search term.
                        ```fill("123", "search term")```  # Use the textbox bid, not the button bid

                        CRITICAL: Do NOT include function calls in your reasoning text. For example, do NOT write "I previously clicked the search button (`click("167")`)" - this will be detected as a multi-action. Instead, just say "I previously clicked the search button" without the function call syntax. Only include the action you want to execute NOW in a code block.

                        """,
            }
        )

        # append past actions (and last error message) if any
        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
                            # History of past actions
                            """,
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\
{action}
""",
                    }
                    for action in self.action_history
                ]
            )

            if obs["last_action_error"]:
                # Log error to console
                rich_logger.error(f"Error: {str(obs['last_action_error'])[:100]}...")

                # Add error to message
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
                                # Error message from last action

                                {obs["last_action_error"]}

                                """,
                    }
                )

        # Inject memory exemplars before asking for next action
        memory_exemplars = load_memory_exemplars()
        if memory_exemplars:
            logger.info("Memory exemplars loaded and injected into prompt")
            user_msgs.append(
                {
                    "type": "text",
                    "text": memory_exemplars,
                }
            )
        else:
            logger.warning("No memory exemplars loaded - check if memory_exemplars.json exists")

        # ask for the next action
        user_msgs.append(
            {
                "type": "text",
                "text": """\
                        # Next action

                        You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page, and the examples above before deciding on your next action.

                        IMPORTANT RULES:
                        1. When you have completed the task and found information that needs to be communicated (such as describing a product, providing search results, or answering a question), you MUST use send_msg_to_user() to send a message describing what you found. Simply displaying information on the page is not enough - you need to explicitly send a message to complete the task. Do not use noop() when you have completed the task and have information to share.
                        2. For RETRIEVAL tasks (tasks that ask you to "retrieve", "list", "find", or "get" information): After navigating to the correct page, you MUST actually extract and include the requested information in your send_msg_to_user() message. Do NOT just say you are "ready to retrieve" or "ready to list" - you must actually read the page content (from the accessibility tree, HTML, or screenshot) and include the specific items, names, details, or data in your message. For example, if asked to "retrieve all product listings", your message must include the actual product names and details, not just a statement that you found them.
                        2a. CRITICAL - Extracting product information: You MUST read the accessibility tree text carefully to find ACTUAL product names, prices, and descriptions. The product information IS in the tree - you need to search for it. Look for:
                            - Text nodes containing product names (often in links, headings, or button names)
                            - Price information (look for $, USD, or numeric values)
                            - Product descriptions or details
                        2b. NEVER make up or guess product names. Do NOT use generic descriptions like "noise-cancelling headphone" or "wireless headphone" unless those exact words appear in the accessibility tree. If you cannot find specific product names in the current view, you may need to scroll down using scroll() to see more products. Only report information that is ACTUALLY visible in the accessibility tree text provided to you.
                        2c. When listing products, include the EXACT names as they appear in the tree, along with any available details like prices or descriptions. Format your message clearly with each product on a separate line or bullet point.
                        3. CRITICAL - Using fill() and press() correctly:
                            - fill() can ONLY be used on elements with role='textbox', role='input', type='text', or type='textarea'. NEVER use fill() on elements with role='button' or type='button'.
                            - When you need to enter text (like in a search field), you MUST find the element with role='textbox' in the accessibility tree. 
                            - HOW TO FIND A TEXTBOX: Search the accessibility tree text for the word "textbox". The tree format shows elements like "[bid] textbox 'name' (description)". Look for lines containing "textbox" and note the bid (the number/ID in brackets). That bid is what you use with fill().
                            - If you click a search button and it opens a search interface, a NEW input field (textbox) will appear in the updated accessibility tree. You MUST search the tree text for "textbox" to find it - do NOT scroll or guess. The textbox will be visible in the tree text if it exists on the page.
                            - If you cannot find "textbox" in the accessibility tree after clicking a search button, the search interface may not have opened yet, or you may need to wait. But first, actually SEARCH the tree text for the word "textbox" - don't just scroll.
                            - press() is for single keys only (like "Enter", "Tab", "Escape"). It CANNOT be used to type text strings. To type text, you MUST use fill() on a textbox element.
                            - Buttons are for clicking ONLY - you CANNOT type text into them. If you get an error saying "Element is not an <input>", it means you're trying to fill a button - find the textbox instead.
                        4. If you receive an error message, you MUST address it. Do not assume an action succeeded if you received an error. Retry the failed action or find an alternative approach.
                        5. Only output ONE action per response. Do not include multiple action calls in a single response, even if they are separated by text. Each action must be in its own separate response.
                        5a. CRITICAL - Do NOT include function calls in your reasoning text. The parser scans your ENTIRE response for function calls, not just code blocks. If you mention a previous action like `click("167")` in your reasoning AND provide a new action, it will be detected as multiple actions and cause an error. Only include the action you want to execute NOW in a code block at the end. Do not reference previous actions using function call syntax in your text.
                        6. Verify that your actions have actually completed successfully before moving on to the next step. If an action fails, you must retry it.
                        """,
            }
        )

        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            match message["type"]:
                case "text":
                    prompt_text_strings.append(message["text"])
                case "image_url":
                    image_url = message["image_url"]
                    if isinstance(message["image_url"], dict):
                        image_url = image_url["url"]
                    if image_url.startswith("data:image"):
                        prompt_text_strings.append(
                            "image_url: " + image_url[:30] + "... (truncated)"
                        )
                    else:
                        prompt_text_strings.append("image_url: " + image_url)
                case _:
                    raise ValueError(
                        f"Unknown message type {repr(message['type'])} in the task goal."
                    )
        "\n".join(prompt_text_strings)
        # Don't log the full prompt - too verbose
        # logger.info(full_prompt_txt)

        # query model using the abstraction function
        action = self.query_model(system_msgs, user_msgs)

        # Extract action type for a cleaner log message
        action_type = action.split("(")[0] if "(" in action else "unknown"
        action_args = action.split("(", 1)[1].rstrip(")") if "(" in action else ""

        # Log concise action summary to console
        step_num = len(self.action_history) + 1
        action_summary = f"{action_type}"
        if action_args:
            action_summary += f"({action_args[:50]}{'...' if len(action_args) > 50 else ''})"

        rich_logger.task_step(step_num, action_summary)

        self.action_history.append(action)

        # Store observation for metrics
        self.update_last_observation(obs)

        return action, {}


@dataclasses.dataclass
class DemoAgentArgs(AbstractAgentArgs):
    model_name: str = "gpt-4o"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False
    system_message_handling: Literal["separate", "combined"] = "separate"
    thinking_budget: int = 10000

    # API keys and configuration - these can be None and the agent will fall back to environment variables
    openai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openrouter_site_url: Optional[str] = None
    openrouter_site_name: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    def make_agent(self):
        return DemoAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            system_message_handling=self.system_message_handling,
            thinking_budget=self.thinking_budget,
            # Pass API keys and configuration
            openai_api_key=self.openai_api_key,
            openrouter_api_key=self.openrouter_api_key,
            openrouter_site_url=self.openrouter_site_url,
            openrouter_site_name=self.openrouter_site_name,
            anthropic_api_key=self.anthropic_api_key,
        )
