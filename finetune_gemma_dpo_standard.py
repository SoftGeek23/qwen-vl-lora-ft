#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) fine-tuning script for Gemma-3-27B-IT using standard TRL.

This script uses standard transformers + TRL + PEFT for DPO training, avoiding Unsloth dependency issues.

Usage:
    python finetune_gemma_dpo_standard.py --dataset_path ./dpo_dataset.jsonl
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

# Set Hugging Face cache to /workspace if available (BEFORE any HF imports)
if os.path.exists("/workspace"):
    hf_cache = "/workspace/.cache/huggingface"
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_DATASETS_CACHE"] = hf_cache

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    import numpy as np
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install: pip install transformers trl peft bitsandbytes datasets")
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DPOFineTuneConfig:
    """Configuration for DPO fine-tuning."""
    # Model config
    model_name: str = "google/gemma-2-27b-it"
    max_seq_length: int = 4096
    
    # LoRA config
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # DPO training config
    beta: float = 0.25
    learning_rate: float = 2e-6
    num_epochs: int = 8
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 20
    warmup_ratio: float = 0.1
    save_steps: int = 25
    logging_steps: int = 5
    eval_steps: int = 25
    output_dir: str = "./gemma_dpo_checkpoint"
    
    # Small dataset optimizations
    dataset_repetitions: int = 3
    validation_split: float = 0.2
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine"
    save_total_limit: int = 3
    
    # Dataset config
    dataset_path: str = "./dpo_dataset.jsonl"
    
    # Optional
    cache_dir: Optional[str] = None
    hf_token: Optional[str] = None
    seed: int = 42


def load_dpo_dataset(jsonl_path: str, repetitions: int = 1) -> Dataset:
    """Load DPO dataset from JSONL file."""
    logger.info(f"Loading DPO dataset from {jsonl_path}")
    
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
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
    
    if repetitions > 1:
        examples = examples * repetitions
        logger.info(f"Repeated dataset {repetitions}x: {original_count} -> {len(examples)} examples")
    
    return Dataset.from_list(examples)


def setup_model_and_tokenizer(config: DPOFineTuneConfig):
    """Setup model and tokenizer for DPO training."""
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
    
    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=token,
        trust_remote_code=True,
    )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Add LoRA adapters
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable %: {100 * trainable_params / all_params:.2f}")
    
    return model, tokenizer


def main():
    """Main DPO training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO fine-tune Gemma-3-27B-IT with TRL")
    parser.add_argument("--dataset_path", type=str, default="./dpo_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./gemma_dpo_checkpoint")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-27b-it")
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dataset_repetitions", type=int, default=3)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    
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
    config = DPOFineTuneConfig(
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
    
    # Set random seed
    import random
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Load dataset
    logger.info("Loading DPO dataset...")
    dataset = load_dpo_dataset(config.dataset_path, repetitions=config.dataset_repetitions)
    
    # Create train/validation split
    if config.validation_split > 0 and len(dataset) > 10:
        dataset = dataset.train_test_split(test_size=config.validation_split, seed=config.seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} validation")
    else:
        train_dataset = dataset
        eval_dataset = None
        logger.info(f"Using full dataset for training: {len(train_dataset)} examples")
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Calculate training steps
    examples_per_step = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, len(train_dataset) // examples_per_step)
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = max(config.warmup_steps, int(total_steps * config.warmup_ratio))
    
    logger.info(f"Training configuration:")
    logger.info(f"  Total examples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"  Validation examples: {len(eval_dataset)}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  DPO beta: {config.beta}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    # DPO Training arguments (using DPOConfig instead of TrainingArguments)
    training_args = DPOConfig(
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size if eval_dataset else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        bf16=True,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.eval_steps if eval_dataset else None,
        save_steps=config.save_steps,
        optim="adamw_torch",
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=config.output_dir,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        report_to="none",
        max_grad_norm=config.max_grad_norm,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # DPO-specific parameters
        beta=config.beta,
        max_length=config.max_seq_length,
        max_prompt_length=config.max_seq_length // 2,
    )
    
    # DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer will create reference model internally
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("Starting DPO training...")
    logger.info("=" * 60)
    
    train_result = dpo_trainer.train()
    
    # Log training metrics
    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    
    # Save model
    logger.info(f"Saving model to {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("✅ DPO training complete!")
    logger.info(f"Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
