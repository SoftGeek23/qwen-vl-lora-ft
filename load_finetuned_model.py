#!/usr/bin/env python3
"""
Helper script to prepare fine-tuned LoRA model for vLLM or merge LoRA weights.

IMPORTANT: LoRA adapters trained on base model can be used with AWQ models for inference!

Usage:
    # Option 1: Verify LoRA adapter works with AWQ model (for vLLM inference)
    python load_finetuned_model.py --lora_path ./qwen2_5_vl_lora_checkpoint --base_model Qwen/Qwen2.5-VL-32B-Instruct-AWQ

    # Option 2: Merge LoRA weights into BASE model (then quantize if needed)
    python load_finetuned_model.py --lora_path ./qwen2_5_vl_lora_checkpoint --base_model Qwen/Qwen2.5-VL-32B-Instruct --merge --output_path ./qwen2_5_vl_merged

    # Option 3: Verify LoRA adapter works with base model (for testing)
    python load_finetuned_model.py --lora_path ./qwen2_5_vl_lora_checkpoint --base_model Qwen/Qwen2.5-VL-32B-Instruct
"""

import argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_lora_weights(lora_path: str, base_model_name: str, output_path: str):
    """
    Merge LoRA weights into base model for easier deployment.
    
    NOTE: Merging should be done with the BASE model (not AWQ), as merging
    quantized models can be problematic. After merging, you can quantize
    the merged model if needed.
    """
    if "awq" in base_model_name.lower():
        logger.error("❌ ERROR: Merging LoRA into AWQ models is not recommended!")
        logger.error("   LoRA adapters should be merged into the base model first.")
        logger.error("   Use: Qwen/Qwen2.5-VL-32B-Instruct (without -AWQ)")
        logger.error("   For AWQ inference, use LoRA adapters directly with vLLM instead.")
        logger.error("   Example: vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ --enable-lora --lora-modules my-lora=<lora_path>")
        raise ValueError("Cannot merge LoRA into AWQ model. Use base model for merging.")
    
    logger.info(f"Loading base model: {base_model_name}")
    dtype = torch.float16 if "awq" in base_model_name.lower() else torch.bfloat16
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
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
    """
    Verify that LoRA adapter can be loaded correctly.
    
    This works with both base and AWQ models. LoRA adapters trained on
    base models are compatible with AWQ models for inference.
    """
    is_awq = "awq" in base_model_name.lower()
    logger.info(f"Loading base model: {base_model_name}")
    if is_awq:
        logger.info("ℹ️  Using AWQ model - LoRA adapters from base model are compatible!")
    
    dtype = torch.float16 if is_awq else torch.bfloat16
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    logger.info("✅ LoRA adapter loaded successfully!")
    logger.info("\n" + "="*60)
    logger.info("To use with vLLM for efficient AWQ inference:")
    logger.info("="*60)
    logger.info(f"  vllm serve {base_model_name} \\")
    logger.info(f"      --port 7999 \\")
    logger.info(f"      --trust-remote-code \\")
    logger.info(f"      --enable-lora \\")
    logger.info(f"      --lora-modules my-lora={lora_path}")
    logger.info("\nThis will use the AWQ model with your fine-tuned LoRA adapters!")


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
        help="Base model name or path. Use AWQ model for inference, base model for merging."
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

