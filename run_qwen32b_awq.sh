#!/bin/bash
# Fine-tuning script for Qwen2.5-VL-32B-Instruct with LoRA
# Note: Using base model (non-AWQ) for fine-tuning as AWQ models don't support training
# After fine-tuning, you can merge LoRA weights into AWQ model for efficient inference

set -e

# Check for AutoAWQ compatibility issue
if pip show autoawq > /dev/null 2>&1; then
    echo "⚠️  WARNING: autoawq is installed. This may cause import errors during fine-tuning."
    echo "   PEFT tries to import AutoAWQ code even for base models."
    echo ""
    echo "   To fix this, run: ./fix_awq_issue.sh"
    echo "   Or manually: pip uninstall autoawq -y"
    echo ""
    echo "   You can reinstall autoawq later for inference."
    echo ""
    echo "   Continuing anyway... (if you get import errors, run ./fix_awq_issue.sh)"
    echo ""
fi

# Model configuration
# Note: Using base model (non-AWQ) for fine-tuning as AWQ models don't support training
# You can merge the LoRA weights back into the AWQ model after fine-tuning for inference
MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"
DATASET_PATH="./finetuning_dataset.jsonl"
OUTPUT_DIR="./qwen2_5_vl_lora_checkpoint"

# Training hyperparameters (optimized for 32B AWQ model)
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4
WARMUP_STEPS=10
MAX_LENGTH=2048

# LoRA configuration
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# Training settings
SAVE_STEPS=50
LOGGING_STEPS=10

echo "=========================================="
echo "Fine-tuning Qwen2.5-VL-32B-Instruct"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA rank: $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  Max length: $MAX_LENGTH"
echo ""
echo "Starting training..."
echo ""

# Run the fine-tuning script
python finetune_qwen_lora.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --max_length "$MAX_LENGTH" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS"

echo ""
echo "=========================================="
echo "Fine-tuning complete!"
echo "=========================================="
echo "Checkpoint saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo ""
echo "1. Verify LoRA adapter works with AWQ model:"
echo "   python load_finetuned_model.py --lora_path $OUTPUT_DIR --base_model Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
echo ""
echo "2. Deploy AWQ model with LoRA adapters (recommended for cost-efficient inference):"
echo "   ./deploy_awq_with_lora.sh --lora_path $OUTPUT_DIR"
echo ""
echo "   Or manually:"
echo "   vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ --port 7999 --trust-remote-code --enable-lora --lora-modules my-lora=$OUTPUT_DIR"
echo ""
echo "See AWQ_INFERENCE_GUIDE.md for detailed instructions."
echo ""

