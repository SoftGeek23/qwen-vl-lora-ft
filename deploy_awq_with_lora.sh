#!/bin/bash
# Deploy AWQ model with LoRA adapters using vLLM
# This script starts vLLM server with your fine-tuned LoRA adapters on the AWQ model

set -e

# Configuration
AWQ_MODEL="Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
LORA_PATH="./qwen2_5_vl_lora_checkpoint"
PORT=7999
LORA_NAME="my-lora"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lora_path)
            LORA_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --lora_name)
            LORA_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--lora_path PATH] [--port PORT] [--lora_name NAME]"
            exit 1
            ;;
    esac
done

# Check if LoRA adapter exists
if [ ! -d "$LORA_PATH" ]; then
    echo "❌ Error: LoRA adapter not found at: $LORA_PATH"
    echo "   Run fine-tuning first: ./run_qwen32b_awq.sh"
    exit 1
fi

echo "=========================================="
echo "Deploying AWQ Model with LoRA Adapters"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  AWQ Model: $AWQ_MODEL"
echo "  LoRA Adapter: $LORA_PATH"
echo "  LoRA Name: $LORA_NAME"
echo "  Port: $PORT"
echo ""
echo "This will:"
echo "  ✅ Use AWQ quantization (4x memory reduction)"
echo "  ✅ Apply your fine-tuned LoRA adapters"
echo "  ✅ Start vLLM server for inference"
echo ""
echo "Starting vLLM server..."
echo ""

# Start vLLM server
vllm serve "$AWQ_MODEL" \
    --port "$PORT" \
    --trust-remote-code \
    --enable-lora \
    --lora-modules "$LORA_NAME=$LORA_PATH"

echo ""
echo "✅ vLLM server started!"
echo "   Model available at: http://localhost:$PORT"
echo "   Your fine-tuned LoRA adapter is active!"

