#!/usr/bin/env python3
"""
Fine-tuning script for Gemma-3-27B-IT using PEFT (LoRA).

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
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from huggingface_hub import login as hf_login
import logging
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded Hugging Face token (set this if token retrieval fails)
HF_TOKEN = ""g


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default="google/gemma-3-27b-it",
        metadata={"help": "Model identifier from Hugging Face"}
    )
    output_dir: str = field(
        default="./gemma_lora_checkpoint",
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
        metadata={"help": "Target modules for LoRA adaptation. For Gemma, these are the attention and MLP layers."}
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


def preprocess_function(examples: Dict, tokenizer: AutoTokenizer, max_length: int = 2048) -> Dict:
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
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize all texts
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


def setup_model_and_tokenizer(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    device_map: str = "auto",
    token: Optional[str] = None
) -> tuple:
    """Setup model, tokenizer, and LoRA configuration."""
    logger.info(f"Loading model: {model_args.model_name}")
    
    # Get token: use provided, then hardcoded, then environment, then HfFolder
    if token is None:
        # Use hardcoded token if available (defined at module level, line 45)
        if HF_TOKEN:
            token = HF_TOKEN
            logger.info("✅ Using hardcoded HF_TOKEN")
    
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            logger.info("✅ Found token from environment variable")
    
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                logger.info(f"✅ Found token from Hugging Face cache")
        except Exception:
            pass
    
    # CRITICAL: Set token as environment variable so transformers can find it
    if token:
        os.environ["HF_TOKEN"] = token
        logger.info("✅ Set HF_TOKEN environment variable")
    else:
        raise ValueError("❌ No Hugging Face token found! Please set HF_TOKEN in the script or environment.")
    
    # Verify authentication
    if token:
        try:
            from huggingface_hub import whoami
            user = whoami(token=token)
            logger.info(f"✅ Authenticated as: {user.get('name', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️  Token verification failed: {e}")
    else:
        # Try to use existing login without explicit token
        try:
            from huggingface_hub import whoami
            user = whoami()
            logger.info(f"✅ Using existing Hugging Face authentication: {user.get('name', 'unknown')}")
        except Exception:
            logger.warning("⚠️  No Hugging Face token found. If model is gated, you may need to set HF_TOKEN environment variable or run 'huggingface-cli login'")
    
    # For gated models, we MUST pass the token explicitly
    # If token is None, transformers might not use the cached token
    load_kwargs = {
        "cache_dir": model_args.cache_dir,
        "trust_remote_code": True,
    }
    if token:
        load_kwargs["token"] = token
    else:
        # Fallback: try use_auth_token for older transformers compatibility
        load_kwargs["use_auth_token"] = True
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        **load_kwargs
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    dtype = torch.bfloat16  # Use bfloat16 for better training stability
    
    logger.info("Loading model for fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        torch_dtype=dtype,
        device_map=device_map,
        **load_kwargs
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
    logger.info("Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    
    # Critical: Enable training mode and ensure LoRA parameters are trainable
    model.train()
    
    # Explicitly enable gradients for ALL LoRA parameters
    # PEFT should handle this automatically, but we ensure it's set correctly
    trainable_count = 0
    for name, param in model.named_parameters():
        # Check if this is a LoRA parameter
        if "lora" in name.lower():
            if not param.requires_grad:
                param.requires_grad = True
                trainable_count += 1
                logger.debug(f"Enabled gradients for: {name}")
        else:
            # Non-LoRA parameters should be frozen
            param.requires_grad = False
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Verify that some parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if trainable_params == 0:
        raise ValueError(
            "No trainable parameters found! LoRA adapters may not be properly configured."
        )
    
    logger.info(f"✅ Found {trainable_params:,} trainable parameters out of {total_params:,} total ({100*trainable_params/total_params:.2f}%)")
    
    return model, tokenizer


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3-27B-IT with LoRA")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./omnizon_strategy_examples.jsonl",
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma_lora_checkpoint",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model name or path"
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
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for accessing gated models. Can also set HF_TOKEN env var."
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
    
    # Setup LoRA arguments
    lora_args = LoraArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args, lora_args, token=args.hf_token)
    
    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Use bfloat16 for Gemma
    use_bf16 = True
    use_fp16 = False
    
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
        fp16=use_fp16,  # Use fp16 for AWQ models (if supported)
        bf16=use_bf16,  # Use bf16 for base models (better training stability)
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Disable to save memory
        gradient_checkpointing=False,  # Disable to avoid gradient issues with LoRA
        max_grad_norm=1.0,  # Gradient clipping
        dataloader_num_workers=0,  # Reduce memory overhead
        max_steps=-1,  # Use epochs instead
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
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
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

