#!/usr/bin/env python3
"""
Generate improved fine-tuning dataset with:
1. Removed problematic examples (function calls in reasoning)
2. Fixed examples to avoid function calls in reasoning text
3. Added many more synthetic examples covering key failure patterns
4. Focus on multi-action prevention (critical issue)
"""

import json

examples = [
    # ========== MULTI-ACTION PREVENTION (CRITICAL) ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSearch for \"laptop\" and click the search button.\n\n# Current page Accessibility Tree\n\n[E1] textbox '' (Search input field)\n[E2] button 'Search' (Search submit button)\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I need to search for laptop. I can see a textbox E1 and a search button E2. I must do this in two separate steps: first fill the textbox, then in the next step click the button. I will only do the first action now.\n\n```python\nfill(\"E1\", \"laptop\")\n```"}
        ],
        "metadata": {"pattern": "single_action_only", "principle": "one_action_per_response", "strategy_level": "critical", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nAdd two products to cart.\n\n# Current page Accessibility Tree\n\n[E1] button 'Add to cart' (Product 1)\n[E2] button 'Add to cart' (Product 2)\n\n# Action History\n\nclick(\"E1\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I successfully added the first product. Now I need to add the second product. I will click on E2. I must only output ONE action - just the click action for the second product.\n\n```python\nclick(\"E2\")\n```"}
        ],
        "metadata": {"pattern": "sequential_actions", "principle": "one_action_per_response", "strategy_level": "critical", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nFill the search box and submit.\n\n# Current page Accessibility Tree\n\n[E1] textbox '' (Search input)\n[E2] button 'Search'\n\n# Action History\n\nfill(\"E1\", \"smartphone\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I have filled the search box with smartphone. Now I need to click the search button to submit. I will only output the click action - no other actions.\n\n```python\nclick(\"E2\")\n```"}
        ],
        "metadata": {"pattern": "single_action_after_fill", "principle": "one_action_per_response", "strategy_level": "critical", "source": "synthetic"}
    },
    
    # ========== DATA EXTRACTION ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nIdentify the most expensive product in the Gaming category.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Gaming Products'\n[E2] link 'PlayStation DualSense Controller - $74.00'\n[E3] link 'Xbox Series X Console - $449.90'\n[E4] link 'HyperX Cloud III - $99.99'\n[E5] link 'SAMSUNG Galaxy S24 Ultra - $1419.99'\n[E6] link 'SAMSUNG Galaxy Z Fold 6 - $2019.99'\n[E7] link 'PlayStation 5 - $498.99'\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I need to identify the most expensive product. I will read the accessibility tree carefully to extract all prices and compare them. Looking at the tree, I can see prices: $74.00, $449.90, $99.99, $1419.99, $2019.99, $498.99. The highest price is $2019.99 for SAMSUNG Galaxy Z Fold 6. I must extract this information and send it to the user.\n\n```python\nsend_msg_to_user(\"The most expensive product in the Gaming category is SAMSUNG Galaxy Z Fold 6 at $2019.99.\")\n```"}
        ],
        "metadata": {"pattern": "data_extraction_comparison", "principle": "read_tree_extract_compare", "strategy_level": "high_level", "source": "omnizon-7_success"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nRetrieve all product listings from the Headphones category.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Headphones'\n[E2] link 'Sony WH-1000XM5 - $399.99'\n[E3] link 'Bose QuietComfort 45 - $329.99'\n[E4] link 'Apple AirPods Pro - $249.99'\n[E5] link 'Sennheiser HD 660S - $499.99'\n[E6] link 'Audio-Technica ATH-M50x - $149.99'\n\n# Action History\n\nclick(\"452\")\nscroll(0, 500)\nscroll(0, 500)\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I've scrolled multiple times and can now see 5 products in the accessibility tree. For retrieval tasks, I must actually extract the information from the tree and include it in my message. I can see: Sony WH-1000XM5 ($399.99), Bose QuietComfort 45 ($329.99), Apple AirPods Pro ($249.99), Sennheiser HD 660S ($499.99), Audio-Technica ATH-M50x ($149.99). I will extract all these and send them to the user.\n\n```python\nsend_msg_to_user(\"I found the following products in the Headphones category:\\n\\n1. Sony WH-1000XM5 - $399.99\\n2. Bose QuietComfort 45 - $329.99\\n3. Apple AirPods Pro - $249.99\\n4. Sennheiser HD 660S - $499.99\\n5. Audio-Technica ATH-M50x - $149.99\")\n```"}
        ],
        "metadata": {"pattern": "extract_after_scrolling", "principle": "read_tree_extract_all_visible_data", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nDisplay the specifications of the KTC 24 inch Gaming Monitor.\n\n# Current page Accessibility Tree\n\n[E1] heading 'KTC 24 inch 1500R Curved Gaming Monitor'\n[E2] text 'Size: 23.8\"'\n[E3] text 'Brand: KTC'\n[E4] text 'Screen Size: 24 Inches'\n[E5] text 'Resolution: FHD'\n[E6] text 'Aspect Ratio: 16:9'\n[E7] text 'Screen Surface: Matte'\n[E8] text 'Price: $119.99'\n\n# Action History\n\nclick(\"171\")\nclick(\"2388\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I'm on the product page and can see the specifications in the accessibility tree. For retrieval tasks, I must actually read the accessibility tree and extract the specific information requested, then send it to the user. I can see: Size: 23.8\", Brand: KTC, Screen Size: 24 Inches, Resolution: FHD, Aspect Ratio: 16:9, Screen Surface: Matte. I will extract and send this information.\n\n```python\nsend_msg_to_user(\"Specifications:\\nSize: 23.8\\\"\\nBrand: KTC\\nScreen Size: 24 Inches\\nResolution: FHD\\nAspect Ratio: 16:9\\nScreen Surface: Matte\")\n```"}
        ],
        "metadata": {"pattern": "extract_product_specs", "principle": "read_tree_extract_specific_data", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== COMBOBOX INTERACTIONS ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nChange the quantity to 5.\n\n# Current page Accessibility Tree\n\n[E1] combobox 'Quantity: 1' (aria-expanded=\"true\", aria-haspopup=\"listbox\")\n[E2] listbox '' (Menu container)\n[E3] option '1' (role=\"option\", aria-selected=\"false\")\n[E4] option '2' (role=\"option\", aria-selected=\"false\")\n[E5] option '3' (role=\"option\", aria-selected=\"false\")\n[E6] option '4' (role=\"option\", aria-selected=\"false\")\n[E7] option '5' (role=\"option\", aria-selected=\"false\")\n[E8] option '6' (role=\"option\", aria-selected=\"false\")\n\n# Action History\n\nclick(\"E1\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I clicked the quantity combobox and it's now expanded. When a combobox is expanded, I can see the menu options in the accessibility tree. I need to click on the specific option I want, not try to use select_option which only works on select elements. The combobox shows options 1-6. I want option 5 which has bid E7. I will click on that option.\n\n```python\nclick(\"E7\")\n```"}
        ],
        "metadata": {"pattern": "combobox_interaction", "principle": "click_combobox_then_click_option", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSet quantity to maximum allowed (30).\n\n# Current page Accessibility Tree\n\n[E1] combobox 'Quantity: 1' (aria-expanded=\"true\", aria-haspopup=\"listbox\")\n[E2] listbox '' (Menu)\n[E3] option '1' (role=\"option\")\n[E4] option '2' (role=\"option\")\n[E5] option '5' (role=\"option\")\n[E6] option '10' (role=\"option\")\n[E7] option '20' (role=\"option\")\n[E8] option '30' (role=\"option\")\n\n# Action History\n\nclick(\"E1\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "The combobox is expanded and I can see the menu options in the accessibility tree. When a combobox is expanded, the menu items appear in the tree with role option. I can see options up to 30 (E8). Since I need the maximum (30), I will click on option E8. I should not try to use select_option on comboboxes - they're not select elements, I must click the option directly.\n\n```python\nclick(\"E8\")\n```"}
        ],
        "metadata": {"pattern": "combobox_option_selection", "principle": "click_option_when_combobox_expanded", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSelect the year 2029 from the expiration date dropdown.\n\n# Current page Accessibility Tree\n\n[E1] combobox '2024' (aria-expanded=\"true\", id=\"year-select\")\n[E2] listbox '' (Menu, role=\"presentation\")\n[E3] backdrop '' (MuiBackdrop, aria-hidden=\"true\")\n[E4] option '2024' (role=\"option\", data-value=\"2024\")\n[E5] option '2025' (role=\"option\", data-value=\"2025\")\n[E6] option '2026' (role=\"option\", data-value=\"2026\")\n[E7] option '2027' (role=\"option\", data-value=\"2027\")\n[E8] option '2028' (role=\"option\", data-value=\"2028\")\n[E9] option '2029' (role=\"option\", data-value=\"2029\")\n\n# Action History\n\nclick(\"E1\")\n\n# Error message from last action\n\nTimeoutError: Locator.click: Timeout 500ms exceeded. <div bid=\"E3\"> backdrop intercepts pointer events\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "The combobox is expanded and I can see the menu options. The error mentions a backdrop intercepting clicks, but this is normal for dropdown menus - the backdrop is there to capture clicks outside. When a combobox is expanded, I can see the menu options in the tree. I should click directly on the option I want (E9 for 2029), not on the combobox again. The backdrop won't block clicks on the menu items themselves.\n\n```python\nclick(\"E9\")\n```"}
        ],
        "metadata": {"pattern": "modal_dropdown_interaction", "principle": "click_menu_option_when_expanded", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSelect delivery date to the last available date.\n\n# Current page Accessibility Tree\n\n[E1] combobox 'Delivery date' (aria-expanded=\"true\")\n[E2] listbox '' (Menu, role=\"presentation\")\n[E3] backdrop '' (MuiBackdrop, intercepts pointer events)\n[E4] option 'Dec 15, 2024' (role=\"option\")\n[E5] option 'Dec 20, 2024' (role=\"option\")\n[E6] option 'Dec 25, 2024' (role=\"option\")\n[E7] option 'Dec 30, 2024' (role=\"option\")\n\n# Action History\n\nclick(\"E1\")\n\n# Error message from last action\n\nTimeoutError: backdrop intercepts pointer events\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "The combobox is expanded and I can see the menu options. The error mentions a backdrop, but this is normal for dropdown menus. The backdrop is there to capture outside clicks, but it does not block clicks on the menu items themselves. I can see the last available date is Dec 30, 2024 (E7). I should click directly on that option - the backdrop won't prevent it.\n\n```python\nclick(\"E7\")\n```"}
        ],
        "metadata": {"pattern": "modal_menu_item_click", "principle": "backdrop_doesnt_block_menu_items", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== SEARCH WORKFLOW ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSearch for \"laptop\".\n\n# Current page Accessibility Tree\n\n[E1] button 'Search' (Search icon)\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I need to search for laptop. When I see only a search button (not a visible textbox), clicking the button will open a search interface with a textbox. I should click the search button first, then in the next step, search the updated accessibility tree for textbox to find the input field.\n\n```python\nclick(\"E1\")\n```"}
        ],
        "metadata": {"pattern": "search_interface_workflow", "principle": "click_search_button_then_find_textbox", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSearch for \"laptop\".\n\n# Current page Accessibility Tree\n\n[E1] button 'Search' (Search icon)\n[E2] textbox '' (Search input field)\n[E3] button 'Search' (Search submit button)\n\n# Action History\n\nclick(\"E1\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "After clicking the search button, a textbox (E2) now appears in the accessibility tree. I must search the tree for textbox to find input fields. Element E2 has role textbox, so I will use fill on that element to enter the search query.\n\n```python\nfill(\"E2\", \"laptop\")\n```"}
        ],
        "metadata": {"pattern": "find_textbox_after_search_click", "principle": "search_tree_for_textbox_after_button_click", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSearch for \"smartphone\".\n\n# Current page Accessibility Tree\n\n[E1] textbox '' (Search input field)\n[E2] button 'Search' (Search submit button)\n\n# Action History\n\nfill(\"E1\", \"smartphone\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I have filled the search box with smartphone. Now I need to click the search button to submit the search. I will only output the click action.\n\n```python\nclick(\"E2\")\n```"}
        ],
        "metadata": {"pattern": "click_search_after_fill", "principle": "sequential_actions_separate", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== ERROR RECOVERY ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSearch for \"KTC 24 inch Gaming Monitor\".\n\n# Current page Accessibility Tree\n\n[E1] button 'Search' (Search icon button)\n[E2] textbox '' (Search input field)\n\n# Action History\n\nclick(\"E1\")\n\n# Error message from last action\n\nError: Locator.clear: Error: Element is not an <input>, <textarea>, <select> or [contenteditable]. Element is a button, not an input field.\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I made an error - I tried to use fill on a button element. When clicking a search button opens a search interface, a new textbox appears in the updated accessibility tree. I must search the tree for textbox to find it. The error shows element E1 is a button. I clicked it, which should have opened a search interface. Now I need to find the textbox in the updated tree. Element E2 has role textbox, so I will use fill on that element.\n\n```python\nfill(\"E2\", \"KTC 24 inch Gaming Monitor\")\n```"}
        ],
        "metadata": {"pattern": "error_recovery_search_interface", "principle": "click_search_button_then_find_textbox", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nAdd product to cart.\n\n# Current page Accessibility Tree\n\n[E1] button 'Add to cart' (Product button)\n\n# Action History\n\nclick(\"E1\")\n\n# Error message from last action\n\nValueError: Could not find element with bid \"E1\"\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I received an error that element E1 could not be found. This could mean the page has changed or the element is no longer available. I should scroll down to see if the button is now in a different location, or search the accessibility tree for keywords like \"cart\" or \"add\" to find the correct element. Let me scroll first to reveal more content.\n\n```python\nscroll(0, 500)\n```"}
        ],
        "metadata": {"pattern": "error_recovery_element_not_found", "principle": "scroll_or_search_alternatives", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nChange quantity to 5.\n\n# Current page Accessibility Tree\n\n[E1] combobox 'Quantity: 1'\n\n# Action History\n\nfill(\"E1\", \"5\")\n\n# Error message from last action\n\nError: Locator.clear: Error: Element is not an <input>, <textarea>, <select> or [contenteditable]. Element is a combobox, not an input field.\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I made an error - I tried to use fill on a combobox element. Comboboxes are not input fields. I need to click the combobox first to expand it, then click on the option I want. I will click the combobox to open the menu.\n\n```python\nclick(\"E1\")\n```"}
        ],
        "metadata": {"pattern": "error_recovery_combobox", "principle": "combobox_requires_click_not_fill", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== COMPLETION RECOGNITION ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nSearch for smartphones, add the first two to cart, view the third product, click Buy Now, and proceed to checkout.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Order placed, thanks!'\n[E2] text 'Your order has been confirmed'\n[E3] text 'Confirmation will be sent to your email'\n\n# Action History\n\nfill(\"160\", \"smartphones\")\nclick(\"164\")\nclick(\"654\")\nclick(\"679\")\nclick(\"681\")\nclick(\"913\")\nclick(\"1113\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "Looking at the accessibility tree, I can see \"Order placed, thanks!\" and \"Your order has been confirmed\". When you see a confirmation page indicating successful completion (like \"Order placed\", \"Thanks\", \"Confirmed\"), the task is complete. I should not continue with noop actions. The goal has been achieved - I searched, added products, viewed the third, clicked Buy Now, and placed the order. The confirmation page confirms success. I will stop here.\n\n```python\nnoop()\n```"}
        ],
        "metadata": {"pattern": "recognize_completion", "principle": "stop_on_confirmation_page", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nPurchase a product and complete checkout.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Order placed, thanks!'\n[E2] text 'Your order has been confirmed'\n[E3] text 'Order number: 12345'\n[E4] text 'Confirmation email sent'\n\n# Action History\n\nclick(\"913\")\nclick(\"1113\")\nnoop()\nnoop()\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "The page shows \"Order placed, thanks!\" and \"Your order has been confirmed\". When you see confirmation messages like \"Order placed\", \"Thanks\", \"Confirmed\", \"Success\", the task is complete. I should not continue with noop actions in a loop. The goal was to purchase and checkout, which is done. I will do one final noop to acknowledge completion, but I recognize the task is finished.\n\n```python\nnoop()\n```"}
        ],
        "metadata": {"pattern": "avoid_noop_loops", "principle": "recognize_completion_stop_repeating", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== ALTERNATIVE APPROACHES ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nComplete checkout after adding items to cart.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Cart'\n[E2] text 'Cart has 2 items'\n[E3] link 'Cart' (Cart icon)\n[E4] text 'Subtotal: $99.98'\n\n# Action History\n\nclick(\"E3\")\nclick(\"E3\")\nclick(\"E3\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I've clicked the cart link multiple times but haven't reached checkout. When repeating the same action doesn't work, I must try alternative approaches. Instead of clicking the cart link again, I should: scroll down to see if checkout button is below, search the accessibility tree for keywords like checkout, proceed, place order, continue, or look for buttons or links I might have missed. Let me scroll down first to reveal more content.\n\n```python\nscroll(0, 600)\n```"}
        ],
        "metadata": {"pattern": "alternative_approach_when_stuck", "principle": "try_scroll_search_alternatives", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nFind and click the checkout button.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Cart'\n[E2] text 'Cart has 2 items'\n[E3] text 'Subtotal: $99.98'\n[E4] button 'Proceed to checkout' (Checkout button)\n\n# Action History\n\nscroll(0, 600)\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "After scrolling, I can now see the checkout button (E4) in the accessibility tree. I will click on it to proceed to checkout.\n\n```python\nclick(\"E4\")\n```"}
        ],
        "metadata": {"pattern": "scroll_reveals_elements", "principle": "scroll_then_find_elements", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== DATA COMPARISON ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nCompare SAMSUNG Galaxy S24 Ultra and SAMSUNG Galaxy Z Fold 6, and buy the better value.\n\n# Current page Accessibility Tree\n\n[E1] heading 'SAMSUNG Galaxy Z Fold 6'\n[E2] text 'Price: $2019.99'\n[E3] text '512GB storage'\n[E4] text 'Folding screen'\n[E5] button 'Buy now'\n\n# Action History\n\nclick(\"735\")\nclick(\"2748\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I'm on the Z Fold 6 page and see price $2019.99. For comparison tasks, I must actually extract and compare the data from both products before deciding. I saw the S24 Ultra earlier but didn't extract its price. I need to go back or search for it again to get its price, then compare: Z Fold 6 is $2019.99. I need to find S24 Ultra's price to determine which is better value. Let me search for it.\n\n```python\nfill(\"164\", \"SAMSUNG Galaxy S24 Ultra\")\n```"}
        ],
        "metadata": {"pattern": "data_comparison_required", "principle": "extract_both_items_before_comparing", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== SCROLL AND EXTRACT ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nRetrieve all product listings from the Headphones category.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Headphones'\n[E2] link 'Sony WH-1000XM5 - $399.99'\n[E3] link 'Bose QuietComfort 45 - $329.99'\n[E4] link 'Apple AirPods Pro - $249.99'\n\n# Action History\n\nclick(\"452\")\nscroll(0, 500)\nscroll(0, 500)\nscroll(0, 500)\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I've scrolled multiple times to reveal more products. After scrolling, I must actually read the accessibility tree to extract the product information. I can see 3 products in the current tree. I should check if there are more by scrolling once more, but then I must extract all visible products from the tree and send them to the user. Let me scroll one more time to ensure I've seen everything, then extract all products.\n\n```python\nscroll(0, 500)\n```"}
        ],
        "metadata": {"pattern": "scroll_and_extract", "principle": "scroll_then_read_tree_extract_data", "strategy_level": "high_level", "source": "synthetic"}
    },
    
    # ========== ADDITIONAL SYNTHETIC EXAMPLES ==========
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nAdd three different products to cart.\n\n# Current page Accessibility Tree\n\n[E1] button 'Add to cart' (Product A)\n[E2] button 'Add to cart' (Product B)\n[E3] button 'Add to cart' (Product C)\n\n# Action History\n\nclick(\"E1\")\nclick(\"E2\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I have successfully added Product A and Product B to the cart. Now I need to add Product C. I will click on E3. I must only output one action.\n\n```python\nclick(\"E3\")\n```"}
        ],
        "metadata": {"pattern": "sequential_cart_additions", "principle": "one_action_per_response", "strategy_level": "critical", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nFill in payment information: name \"John Doe\", card \"1234 5678 9012 3456\", exp \"12/2025\", CVV \"123\".\n\n# Current page Accessibility Tree\n\n[E1] textbox 'Name on card'\n[E2] textbox 'Card number'\n[E3] textbox 'Expiration date'\n[E4] textbox 'CVV'\n\n# Action History\n\nfill(\"E1\", \"John Doe\")\nfill(\"E2\", \"1234 5678 9012 3456\")\nfill(\"E3\", \"12/2025\")\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I have filled in the name, card number, and expiration date. Now I need to fill in the CVV field. I will fill E4 with the CVV value. Only one action.\n\n```python\nfill(\"E4\", \"123\")\n```"}
        ],
        "metadata": {"pattern": "sequential_form_filling", "principle": "one_action_per_response", "strategy_level": "critical", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nFind the cheapest product in the Electronics category.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Electronics'\n[E2] link 'Product A - $99.99'\n[E3] link 'Product B - $149.99'\n[E4] link 'Product C - $79.99'\n[E5] link 'Product D - $199.99'\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I need to find the cheapest product. I will read the accessibility tree carefully to extract all prices and compare them. Looking at the tree, I can see prices: $99.99, $149.99, $79.99, $199.99. The lowest price is $79.99 for Product C. I must extract this information and send it to the user.\n\n```python\nsend_msg_to_user(\"The cheapest product in the Electronics category is Product C at $79.99.\")\n```"}
        ],
        "metadata": {"pattern": "price_comparison", "principle": "read_tree_extract_compare", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nView product details for \"Wireless Mouse\".\n\n# Current page Accessibility Tree\n\n[E1] link 'Wireless Mouse - $29.99'\n[E2] link 'View details'\n[E3] button 'Add to cart'\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I need to view the product details for Wireless Mouse. I can see a \"View details\" link (E2) associated with the product. I will click on that link to view the product details page.\n\n```python\nclick(\"E2\")\n```"}
        ],
        "metadata": {"pattern": "view_product_details", "principle": "click_view_details_link", "strategy_level": "high_level", "source": "synthetic"}
    },
    {
        "messages": [
            {"role": "system", "content": "# Instructions\n\nReview the current state of the page and all other information to find the best\npossible next action to accomplish your goal. Your answer will be interpreted\nand executed by a program, make sure to follow the formatting instructions."},
            {"role": "user", "content": "# Goal\n\nPlace an order for the items in cart.\n\n# Current page Accessibility Tree\n\n[E1] heading 'Checkout'\n[E2] text 'Cart has 2 items'\n[E3] text 'Total: $199.98'\n[E4] button 'Place your order'\n\n# Next action\n\nYou will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page before deciding on your next action."},
            {"role": "assistant", "content": "I'm on the checkout page and can see the \"Place your order\" button (E4). I will click on it to complete the order.\n\n```python\nclick(\"E4\")\n```"}
        ],
        "metadata": {"pattern": "place_order", "principle": "click_place_order_button", "strategy_level": "high_level", "source": "synthetic"}
    },
]

def main():
    """Generate improved strategy examples."""
    output_file = "omnizon_strategy_examples_improved.jsonl"
    
    print(f"Generating {len(examples)} improved strategy examples...")
    print("\nKey improvements:")
    print("  ✅ Removed function calls from reasoning text")
    print("  ✅ Added explicit multi-action prevention examples")
    print("  ✅ Expanded coverage of critical patterns")
    print("  ✅ More diverse synthetic examples")
    print(f"\nTotal examples: {len(examples)}")
    print("\nPatterns covered:")
    patterns = {}
    for ex in examples:
        pattern = ex['metadata'].get('pattern', 'unknown')
        patterns[pattern] = patterns.get(pattern, 0) + 1
    
    for pattern, count in sorted(patterns.items()):
        print(f"  - {pattern}: {count} example(s)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Created {output_file} with {len(examples)} examples")
    print(f"\nTo use this dataset:")
    print(f"  python finetune_qwen_lora.py --dataset_path {output_file}")

if __name__ == "__main__":
    main()

