#!/usr/bin/env python3
"""
Script to generate training examples for the important rules from basic_agent.py

These examples teach the model to:
1. Use send_msg_to_user() when task is complete
2. Retry failed actions instead of assuming success
3. Only output ONE action per response
4. Verify actions completed before moving on
"""

import json

# Examples for each important rule
examples = [
    # Rule 1: Use send_msg_to_user() when task is complete
    {
        "messages": [
            {
                "role": "system",
                "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."
            },
            {
                "role": "user",
                "content": "# Goal\n\nSearch for \"laptop\" using the search bar and display the first product in the results.\n\n# Current page Accessibility Tree\n\n[E1] heading 'VILVA Portable-Monitor-for-Laptop - 15.6\" 1080P FHD IPS Display USB-C HDMI Portable Monitor for Laptop'\n[E2] text 'Price: $199.99'\n[E3] text 'Description: Portable monitor with USB-C connectivity'\n\n# Action History\n\nfill('163', 'laptop')\nclick('167')\nclick('634')\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action.\n\nIMPORTANT RULES:\n1. When you have completed the task and found information that needs to be communicated (such as describing a product, providing search results, or answering a question), you MUST use send_msg_to_user() to send a message describing what you found. Simply displaying information on the page is not enough - you need to explicitly send a message to complete the task. Do not use noop() when you have completed the task and have information to share."
            },
            {
                "role": "assistant",
                "content": "I have successfully completed the task. I searched for \"laptop\" and found the first product: VILVA Portable-Monitor-for-Laptop. The product details are now displayed on the page. According to the rules, I must use send_msg_to_user() to communicate this information to complete the task. I will send a message describing the product.\n\n```python\nsend_msg_to_user(\"I found the first product in the search results: VILVA Portable-Monitor-for-Laptop - 15.6\\\" 1080P FHD IPS Display USB-C HDMI Portable Monitor for Laptop with Cover and Speakers. The product is priced at $199.99 and features USB-C connectivity.\")\n```"
            }
        ],
        "metadata": {
            "pattern": "send_message_on_completion",
            "principle": "communicate_results_via_message",
            "rule": "rule_1"
        }
    },
    
    # Rule 2: Retry failed actions
    {
        "messages": [
            {
                "role": "system",
                "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."
            },
            {
                "role": "user",
                "content": "# Goal\n\nAdd the first two products to your cart.\n\n# Current page Accessibility Tree\n\n[E1] button 'Add to cart' (Product 1)\n[E2] button 'Add to cart' (Product 2)\n\n# Action History\n\nclick('E1')\n\n# Error message from last action\n\nValueError: Received a multi-action, only single-actions are allowed.\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action.\n\nIMPORTANT RULES:\n2. If you receive an error message, you MUST address it. Do not assume an action succeeded if you received an error. Retry the failed action or find an alternative approach."
            },
            {
                "role": "assistant",
                "content": "I received an error when trying to add the first product to the cart. The error indicates I tried to execute multiple actions in one step. According to the rules, I must address errors and retry failed actions. I will retry adding the first product to the cart, ensuring I only output ONE action.\n\n```python\nclick('E1')\n```"
            }
        ],
        "metadata": {
            "pattern": "error_recovery_retry",
            "principle": "retry_failed_actions",
            "rule": "rule_2"
        }
    },
    
    # Rule 3: Only ONE action per response
    {
        "messages": [
            {
                "role": "system",
                "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."
            },
            {
                "role": "user",
                "content": "# Goal\n\nFill a form with name and email, then submit.\n\n# Current page Accessibility Tree\n\n[E1] textbox '' (Name field)\n[E2] textbox '' (Email field)\n[E3] button 'Submit'\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action.\n\nIMPORTANT RULES:\n3. Only output ONE action per response. Do not include multiple action calls in a single response, even if they are separated by text. Each action must be in its own separate response."
            },
            {
                "role": "assistant",
                "content": "The goal requires filling two fields and submitting. However, I can only execute ONE action per step. I must break this into sequential steps. I will start with the first action: filling the name field.\n\n```python\nfill('E1', 'John Doe')\n```"
            }
        ],
        "metadata": {
            "pattern": "single_action_enforcement",
            "principle": "one_action_per_response",
            "rule": "rule_3"
        }
    },
    
    # Rule 4: Verify actions completed
    {
        "messages": [
            {
                "role": "system",
                "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."
            },
            {
                "role": "user",
                "content": "# Goal\n\nAdd three products to your cart.\n\n# Current page Accessibility Tree\n\n[E1] button 'Add to cart' (Product 1)\n[E2] button 'Add to cart' (Product 2)\n[E3] button 'Add to cart' (Product 3)\n[E4] text 'Cart: 1 item'\n\n# Action History\n\nclick('E1')\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action.\n\nIMPORTANT RULES:\n4. Verify that your actions have actually completed successfully before moving on to the next step. If an action fails, you must retry it."
            },
            {
                "role": "assistant",
                "content": "I clicked the 'Add to cart' button for Product 1. I can see from the page state that the cart now shows '1 item', which confirms the first product was successfully added. Now I can proceed to add the second product. I will execute one action: clicking the 'Add to cart' button for Product 2.\n\n```python\nclick('E2')\n```"
            }
        ],
        "metadata": {
            "pattern": "verify_before_proceeding",
            "principle": "confirm_action_success",
            "rule": "rule_4"
        }
    },
    
    # Combined example: Error recovery + verification
    {
        "messages": [
            {
                "role": "system",
                "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."
            },
            {
                "role": "user",
                "content": "# Goal\n\nAdd two products to your cart.\n\n# Current page Accessibility Tree\n\n[E1] button 'Add to cart' (Product 1)\n[E2] button 'Add to cart' (Product 2)\n[E3] text 'Cart: 0 items'\n\n# Action History\n\nclick('E1')\n\n# Error message from last action\n\nValueError: Received a multi-action, only single-actions are allowed.\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action.\n\nIMPORTANT RULES:\n2. If you receive an error message, you MUST address it. Do not assume an action succeeded if you received an error. Retry the failed action or find an alternative approach.\n4. Verify that your actions have actually completed successfully before moving on to the next step. If an action fails, you must retry it."
            },
            {
                "role": "assistant",
                "content": "I received an error when trying to add Product 1 to the cart. The cart still shows '0 items', which confirms the action did not succeed. According to the rules, I must retry the failed action and verify it completes successfully. I will retry adding Product 1 to the cart, ensuring I only output ONE action.\n\n```python\nclick('E1')\n```"
            }
        ],
        "metadata": {
            "pattern": "error_recovery_with_verification",
            "principle": "retry_and_verify",
            "rule": "rule_2_and_4"
        }
    },
]

def main():
    """Add examples to the training dataset."""
    output_file = "important_rules_examples.jsonl"
    
    print(f"Generating {len(examples)} training examples for important rules...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {output_file} with {len(examples)} examples")
    print("\nTo add these to your training dataset:")
    print(f"  cat {output_file} >> finetuning_dataset.jsonl")
    print("\nOr use them separately:")
    print(f"  python finetune_qwen_lora.py --dataset_path {output_file} ...")

if __name__ == "__main__":
    main()

