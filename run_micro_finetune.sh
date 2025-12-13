#!/bin/bash
# Quick-start script for micro fine-tuning with combined datasets

set -e

echo "🔧 Micro Fine-Tuning Setup"
echo "=========================="
echo ""

# Step 1: Use the high-quality omnizon strategy examples
DATASET="omnizon_strategy_examples.jsonl"
echo "📦 Using omnizon strategy examples..."
COUNT=$(wc -l < "$DATASET")

echo "   Examples: $COUNT"
echo "   Quality: High (based on real task failures)"
echo "   Focus: High-level strategies (combobox, modals, data extraction)"
echo ""

if [ "$COUNT" -lt 15 ]; then
    echo "⚠️  Warning: Only $COUNT examples. Expected 15."
    echo ""
fi

# Step 2: Run fine-tuning
echo "🚀 Starting micro fine-tuning..."
echo ""

python finetune_qwen_lora.py \
    --dataset_path "$DATASET" \
    --output_dir ./gemma_lora_micro \
    --model_name google/gemma-3-27b-it \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_steps 5 \
    --max_length 2048

echo ""
echo "✅ Fine-tuning complete!"
echo ""
echo "📁 Output directory: ./gemma_lora_micro"
echo ""
echo "To load the model:"
echo "  from peft import PeftModel"
echo "  model = PeftModel.from_pretrained(base_model, './gemma_lora_micro')"

