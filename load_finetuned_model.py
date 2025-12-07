#!/usr/bin/env python3
"""
Helper script to prepare fine-tuned LoRA model for vLLM or merge LoRA weights.

Usage:
    # Option 1: Merge LoRA weights into base model (for easier vLLM usage)
    python load_finetuned_model.py --lora_path ./qwen2_5_vl_lora_checkpoint --merge --output_path ./qwen2_5_vl_merged

    # Option 2: Just verify LoRA adapter can be loaded
    python load_finetuned_model.py --lora_path ./qwen2_5_vl_lora_checkpoint
"""

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(lora_path: str, base_model_name: str, output_path: str):
    """Merge LoRA weights into base model for easier deployment."""
    logger.info(f"Loading base model: {base_model_name}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Also save processor
    processor = AutoProcessor.from_pretrained(base_model_name)
    processor.save_pretrained(output_path)
    
    logger.info("✅ Merged model saved successfully!")
    logger.info(f"You can now use this model with vLLM: vllm serve {output_path}")


def verify_lora_adapter(lora_path: str, base_model_name: str):
    """Verify that LoRA adapter can be loaded correctly."""
    logger.info(f"Loading base model: {base_model_name}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    logger.info("✅ LoRA adapter loaded successfully!")
    logger.info("\nTo use with vLLM, use:")
    logger.info(f"  vllm serve {base_model_name} --enable-lora --lora-modules my-lora={lora_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuned LoRA model for vLLM")
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        help="Base model name or path"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA weights into base model (creates standalone model)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./qwen2_5_vl_merged",
        help="Output path for merged model (only used with --merge)"
    )
    
    args = parser.parse_args()
    
    if args.merge:
        merge_lora_weights(args.lora_path, args.base_model, args.output_path)
    else:
        verify_lora_adapter(args.lora_path, args.base_model)


if __name__ == "__main__":
    main()

