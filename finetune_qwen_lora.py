#!/usr/bin/env python3
"""
Fine-tuning script for Qwen2.5-VL-32B-Instruct-AWQ using PEFT (LoRA).

This script fine-tunes the model on instruction-following examples while
avoiding catastrophic forgetting through parameter-efficient fine-tuning.
"""

import json
import os

# Set Hugging Face cache to /workspace if available (BEFORE any HF imports)
if os.path.exists("/workspace"):
    hf_cache = "/workspace/.cache/huggingface"
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_DATASETS_CACHE"] = hf_cache

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from qwen_vl_utils import process_vision_info
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        metadata={"help": "Model identifier from Hugging Face"}
    )
    output_dir: str = field(
        default="./qwen2_5_vl_lora_checkpoint",
        metadata={"help": "Output directory for the fine-tuned model"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for model files"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank (r). Lower values = fewer parameters, less risk of overfitting"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha. Typically 2x lora_r for better scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA adaptation"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type: 'none', 'all', or 'lora_only'"}
    )


def load_dataset(jsonl_path: str) -> Dataset:
    """Load dataset from JSONL file."""
    logger.info(f"Loading dataset from {jsonl_path}")
    
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(examples)} examples")
    return Dataset.from_list(examples)


def preprocess_function(examples: Dict, processor: AutoProcessor, max_length: int = 2048) -> Dict:
    """
    Preprocess examples for training.
    
    The dataset already has messages in the correct format.
    We'll apply the chat template and tokenize.
    """
    # Extract messages from examples
    messages_list = examples["messages"]
    
    # Apply chat template to each conversation
    texts = []
    for messages in messages_list:
        # Use apply_chat_template to format the conversation
        # This handles system/user/assistant roles correctly
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize all texts
    # Note: Qwen2.5-VL processor uses tokenizer for text
    tokenizer = processor.tokenizer
    
    model_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Create labels for causal language modeling
    # Labels are the same as input_ids, but we mask out padding tokens
    labels = model_inputs["input_ids"].clone()
    
    # Set padding tokens to -100 (ignored in loss calculation)
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    
    model_inputs["labels"] = labels
    
    return model_inputs


def setup_model_and_processor(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    device_map: str = "auto"
) -> tuple:
    """Setup model, processor, and LoRA configuration."""
    logger.info(f"Loading model: {model_args.model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        cache_dir=model_args.cache_dir
    )
    
    # Load model with AWQ quantization
    # AWQ models need special handling - they're already quantized
    # Note: AWQ models can be tricky for fine-tuning, but LoRA should work
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.float16,  # AWQ models work better with float16
        device_map=device_map,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules,
        bias=lora_args.bias,
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Critical: Enable training mode and ensure LoRA parameters are trainable
    model.train()
    
    # Explicitly enable gradients for LoRA parameters
    # This is necessary for AWQ models
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            logger.debug(f"Enabled gradients for: {name}")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Verify that some parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if trainable_params == 0:
        raise ValueError(
            "No trainable parameters found! LoRA adapters may not be properly configured. "
            "This can happen with AWQ models - you may need to use the non-quantized base model for fine-tuning."
        )
    
    logger.info(f"✅ Found {trainable_params:,} trainable parameters out of {total_params:,} total ({100*trainable_params/total_params:.2f}%)")
    
    return model, processor


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with LoRA")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./finetuning_dataset.jsonl",
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen2_5_vl_lora_checkpoint",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        help="Model name or path. Note: AWQ models may not support fine-tuning. Use base model 'Qwen/Qwen2.5-VL-32B-Instruct' if you encounter gradient issues."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model files. Default: /workspace/.cache/huggingface (if /workspace exists) or ~/.cache/huggingface"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size (use 1-2 for 32B model)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Warmup steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    
    # Determine cache directory - use /workspace if available (more space)
    if args.cache_dir is None:
        if os.path.exists("/workspace"):
            cache_dir = "/workspace/.cache/huggingface"
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using /workspace for model cache: {cache_dir}")
        else:
            cache_dir = None
            logger.info("Using default Hugging Face cache location")
    else:
        cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using specified cache directory: {cache_dir}")
    
    # Setup model arguments
    model_args = ModelArguments(
        model_name=args.model_name,
        output_dir=args.output_dir,
        cache_dir=cache_dir,
    )
    
    # Determine cache directory - use /workspace if available (more space)
    if args.cache_dir is None:
        if os.path.exists("/workspace"):
            cache_dir = "/workspace/.cache/huggingface"
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using /workspace for model cache: {cache_dir}")
        else:
            cache_dir = None
            logger.info("Using default Hugging Face cache location")
    else:
        cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using specified cache directory: {cache_dir}")
    
    # Setup model arguments
    model_args = ModelArguments(
        model_name=args.model_name,
        output_dir=args.output_dir,
        cache_dir=cache_dir,
    )
    
    # Setup LoRA arguments
    lora_args = LoraArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Setup model and processor
    model, processor = setup_model_and_processor(model_args, lora_args)
    
    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, processor, max_length=args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints
        fp16=True,  # Use float16 for AWQ models
        bf16=False,  # AWQ models may not support bfloat16 training
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Disable to save memory
        gradient_checkpointing=True,  # Save memory
        max_grad_norm=1.0,  # Gradient clipping
        dataloader_num_workers=0,  # Reduce memory overhead
        max_steps=-1,  # Use epochs instead
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

