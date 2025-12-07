#!/bin/bash
# Quick start script for fine-tuning Qwen2.5-VL with LoRA

set -e

# Default values
DATASET_PATH="./finetuning_dataset.jsonl"
OUTPUT_DIR="./qwen2_5_vl_lora_checkpoint"
NUM_EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=4
LEARNING_RATE=2e-4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation)
            GRADIENT_ACCUMULATION="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset_path PATH] [--output_dir DIR] [--num_epochs N] [--batch_size N] [--gradient_accumulation N] [--learning_rate LR]"
            exit 1
            ;;
    esac
done

echo "Starting fine-tuning with:"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning rate: $LEARNING_RATE"
echo ""

python finetune_qwen_lora.py \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE"

echo ""
echo "Fine-tuning complete! Check $OUTPUT_DIR for the trained model."

