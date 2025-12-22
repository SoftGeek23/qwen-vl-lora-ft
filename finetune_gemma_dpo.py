#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) fine-tuning script for Gemma-3-27B-IT using Unsloth.

This script fine-tunes the model using DPO to learn from preference pairs (chosen vs rejected responses),
teaching the model to prefer correct, generalizable strategies over failure patterns.

OPTIMIZED FOR SMALL DATASETS (50 samples):
- Dataset repetition (3x) to simulate larger dataset
- Higher DPO beta (0.25) for stronger preference signal
- More epochs (8) with lower learning rate (2e-6) to prevent overfitting
- Higher dropout (0.1) for regularization
- Validation split (20%) for monitoring
- Gradient accumulation (8) for stable gradients
- Cosine learning rate decay for smooth training

Based on Unsloth's GRPO support for Gemma 3:
- 2x faster training
- 80% less memory usage
- Optimized for Gemma 3 architecture


python finetune_gemma_dpo.py --dataset_path ./dpo_dataset.jsonl
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Set Hugging Face cache to /workspace if available (BEFORE any HF imports)
if os.path.exists("/workspace"):
    hf_cache = "/workspace/.cache/huggingface"
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_DATASETS_CACHE"] = hf_cache

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.trainer import UnslothTrainer
    from trl import DPOTrainer
    from transformers import TrainingArguments
    import torch
    import numpy as np
    from datasets import Dataset
    import logging
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install unsloth[colab-new] trl")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO fine-tuning optimized for small datasets (50 samples)."""
    # Model config
    model_name: str = "unsloth/gemma-3-27b-it"
    max_seq_length: int = 4096  # Gemma 3 supports up to 128K, but we use smaller for efficiency
    
    # LoRA config (optimized for small dataset)
    lora_r: int = 32  # Higher rank for DPO (more capacity to learn patterns)
    lora_alpha: int = 64  # 2x rank
    lora_dropout: float = 0.1  # Higher dropout to prevent overfitting on small dataset
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # DPO training config (optimized for 50 samples)
    beta: float = 0.25  # Higher beta for stronger preference signal (critical for small datasets)
    learning_rate: float = 2e-6  # Lower LR to prevent overfitting, more stable
    num_epochs: int = 8  # More epochs since dataset is small (can afford it)
    batch_size: int = 1  # Smallest batch for better gradient estimates
    gradient_accumulation_steps: int = 8  # Larger accumulation to simulate batch_size=8
    warmup_steps: int = 20  # More warmup for stability with small dataset
    warmup_ratio: float = 0.1  # 10% of steps for warmup
    save_steps: int = 25  # Save more frequently for small dataset
    logging_steps: int = 5  # Log more frequently to monitor training
    eval_steps: int = 25  # Evaluate periodically
    output_dir: str = "./gemma_dpo_checkpoint"
    
    # Small dataset optimizations
    dataset_repetitions: int = 3  # Repeat dataset 3x to simulate 150 samples
    validation_split: float = 0.2  # 20% for validation (10 samples)
    weight_decay: float = 0.01  # Regularization to prevent overfitting
    max_grad_norm: float = 0.5  # Gradient clipping for stability
    lr_scheduler_type: str = "cosine"  # Cosine decay for smooth learning
    save_total_limit: int = 3  # Keep only best 3 checkpoints
    
    # Dataset config
    dataset_path: str = "./dpo_dataset.jsonl"
    
    # Optional
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    seed: int = 42  # For reproducibility


def load_dpo_dataset(jsonl_path: str, repetitions: int = 1) -> Dataset:
    """
    Load DPO dataset from JSONL file.
    
    For small datasets, we can repeat the data multiple times to simulate
    a larger dataset and improve training stability.
    
    Expected format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
    """
    logger.info(f"Loading DPO dataset from {jsonl_path}")
    
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                # Validate format
                if not all(key in example for key in ["prompt", "chosen", "rejected"]):
                    logger.warning(f"Line {line_num}: Missing required keys. Skipping.")
                    continue
                examples.append(example)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON. Skipping. Error: {e}")
                continue
    
    original_count = len(examples)
    logger.info(f"Loaded {original_count} DPO examples")
    
    if original_count == 0:
        raise ValueError(f"No valid examples found in {jsonl_path}")
    
    # Repeat dataset to simulate larger dataset (helps with small datasets)
    if repetitions > 1:
        examples = examples * repetitions
        logger.info(f"Repeated dataset {repetitions}x: {original_count} -> {len(examples)} examples")
    
    return Dataset.from_list(examples)




def setup_model_and_tokenizer(config: DPOConfig):
    """Setup Unsloth model and tokenizer for DPO training."""
    logger.info(f"Loading model: {config.model_name}")
    
    # Get token
    token = config.hf_token
    if token is None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token is None:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except Exception:
            pass
    
    if token:
        os.environ["HF_TOKEN"] = token
        logger.info("✅ Using Hugging Face token")
    else:
        logger.warning("⚠️  No Hugging Face token found. Model may be gated.")
    
    # Determine dtype
    dtype = None
    if is_bfloat16_supported():
        dtype = torch.bfloat16
        logger.info("Using bfloat16")
    else:
        dtype = torch.float16
        logger.info("Using float16 (bfloat16 not supported)")
    
    # Load model with Unsloth (optimized for Gemma 3)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=dtype,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        token=token,
        trust_remote_code=True,
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # Left padding for generation
    
    # Enable LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,  # Memory efficient
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main DPO training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO fine-tune Gemma-3-27B-IT with Unsloth")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dpo_dataset.jsonl",
        help="Path to DPO dataset JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gemma_dpo_checkpoint",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-3-27b-it",
        help="Model name (use unsloth/ prefix for optimized version)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="DPO beta parameter (temperature, higher = stronger preference signal). Default 0.25 for small datasets."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Learning rate for DPO training. Lower for small datasets to prevent overfitting."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=8,
        help="Number of training epochs. More epochs for small datasets."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size. Smaller for better gradient estimates with small datasets."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps. Larger to simulate bigger batch size."
    )
    parser.add_argument(
        "--dataset_repetitions",
        type=int,
        default=3,
        help="Number of times to repeat the dataset (helps with small datasets). Default: 3"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Fraction of dataset to use for validation. Default: 0.2 (20%%)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate. Higher for small datasets to prevent overfitting. Default: 0.1"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (higher for DPO)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha (typically 2x rank)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model files"
    )
    
    args = parser.parse_args()
    
    # Determine cache directory
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
    
    # Create config
    global config
    config = DPOConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        dataset_repetitions=args.dataset_repetitions,
        validation_split=args.validation_split,
        hf_token=args.hf_token,
        cache_dir=cache_dir,
    )
    
    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Load dataset with repetitions
    logger.info("Loading DPO dataset...")
    dataset = load_dpo_dataset(config.dataset_path, repetitions=config.dataset_repetitions)
    
    # Create train/validation split for small dataset
    if config.validation_split > 0 and len(dataset) > 10:
        dataset = dataset.train_test_split(
            test_size=config.validation_split,
            seed=config.seed
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} validation")
    else:
        train_dataset = dataset
        eval_dataset = None
        logger.info(f"Using full dataset for training: {len(train_dataset)} examples")
        logger.warning("No validation split (dataset too small). Consider reducing validation_split.")
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Format dataset for DPO
    logger.info("Formatting dataset for DPO training...")
    # DPOTrainer expects prompt/chosen/rejected format
    # The dataset already has this format, but we need to ensure it's compatible
    # with Gemma 3's chat template
    
    def format_dpo_prompt(example):
        """Format a single DPO example for Gemma 3 chat template."""
        prompt_text = example["prompt"]
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]
        
        # Format prompt as user message (Gemma 3 chat format)
        # The prompt already contains the task goal, so we use it directly
        # DPOTrainer will handle the chat template internally
        
        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }
    
    # Apply formatting (dataset already has correct keys, just ensure format)
    formatted_train_dataset = train_dataset.map(
        format_dpo_prompt,
        remove_columns=[col for col in train_dataset.column_names if col not in ["prompt", "chosen", "rejected"]],
    )
    
    if eval_dataset:
        formatted_eval_dataset = eval_dataset.map(
            format_dpo_prompt,
            remove_columns=[col for col in eval_dataset.column_names if col not in ["prompt", "chosen", "rejected"]],
        )
    else:
        formatted_eval_dataset = None
    
    # Calculate total steps for better scheduling (use formatted dataset length)
    examples_per_step = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, len(formatted_train_dataset) // examples_per_step)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = max(config.warmup_steps, int(total_steps * config.warmup_ratio))
    
    logger.info(f"Training configuration:")
    logger.info(f"  Total examples: {len(formatted_train_dataset)}")
    if formatted_eval_dataset:
        logger.info(f"  Validation examples: {len(formatted_eval_dataset)}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps} ({100*warmup_steps/total_steps:.1f}% of training)")
    logger.info(f"  DPO beta: {config.beta} (higher = stronger preference signal)")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    # Training arguments (optimized for small dataset)
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size if eval_dataset else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        save_steps=config.save_steps,
        optim="adamw_torch",
        weight_decay=config.weight_decay,  # Regularization
        lr_scheduler_type=config.lr_scheduler_type,  # Cosine decay
        seed=config.seed,
        output_dir=config.output_dir,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True if eval_dataset else False,  # Save best model
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,  # Lower loss is better
        report_to="none",  # Disable wandb/tensorboard
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
        max_target_length=config.max_seq_length // 2,
        max_grad_norm=config.max_grad_norm,  # Gradient clipping
        dataloader_num_workers=0,  # Reduce memory overhead
        dataloader_pin_memory=False,  # Reduce memory
        remove_unused_columns=False,
    )
    
    # Create reference model for DPO (frozen copy of base model)
    # DPO requires a reference model to compute KL divergence
    logger.info("Creating reference model for DPO...")
    ref_model = FastLanguageModel.get_peft_model(
        model,  # Clone the base model
        r=config.lora_r,
        target_modules=config.lora_target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=False,  # Reference model doesn't need gradients
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=config.beta,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
        max_target_length=config.max_seq_length // 2,
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("Starting DPO training (optimized for small dataset)...")
    logger.info("=" * 60)
    logger.info(f"Training examples: {len(formatted_train_dataset)}")
    if formatted_eval_dataset:
        logger.info(f"Validation examples: {len(formatted_eval_dataset)}")
    logger.info(f"DPO beta: {config.beta} (higher = stronger preference signal)")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info(f"Dropout: {config.lora_dropout} (higher to prevent overfitting)")
    logger.info("=" * 60)
    
    # Train with progress tracking
    train_result = dpo_trainer.train()
    
    # Log training metrics
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    if hasattr(train_result, 'metrics'):
        logger.info(f"Training metrics: {train_result.metrics}")
    
    # Save model
    logger.info(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    # Optionally merge and save for inference
    logger.info("Merging LoRA adapters for inference...")
    model.save_pretrained_merged(
        config.output_dir + "_merged",
        tokenizer,
        save_method="merged_16bit",  # or "merged_4bit" for quantization
    )
    
    logger.info("✅ DPO training complete!")
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info(f"Merged model saved to: {config.output_dir}_merged")


if __name__ == "__main__":
    main()

