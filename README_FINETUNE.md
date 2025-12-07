# Fine-tuning Qwen2.5-VL-32B-Instruct-AWQ with LoRA

This guide explains how to fine-tune the Qwen2.5-VL model using PEFT (LoRA) to avoid catastrophic forgetting.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements_finetune.txt
   ```

2. **Install transformers from source** (required for Qwen2.5-VL):
   ```bash
   pip install git+https://github.com/huggingface/transformers accelerate
   ```

3. **GPU Requirements:**
   - Minimum: 1x A100 (40GB) or equivalent
   - Recommended: 2x A100 (80GB) for faster training
   - The AWQ quantized model requires less memory than the full model

## Dataset Format

The dataset should be in JSONL format with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {...}
}
```

See `finetuning_dataset.jsonl` for examples.

## Training

### Basic Training

```bash
python finetune_qwen_lora.py \
    --dataset_path ./finetuning_dataset.jsonl \
    --output_dir ./qwen2_5_vl_lora_checkpoint \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
```

### Advanced Options

```bash
python finetune_qwen_lora.py \
    --dataset_path ./finetuning_dataset.jsonl \
    --output_dir ./qwen2_5_vl_lora_checkpoint \
    --model_name Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --warmup_steps 10 \
    --max_length 2048
```

## LoRA Configuration

### Key Parameters

- **`lora_r`** (default: 16): LoRA rank. Lower values = fewer parameters, less risk of overfitting
  - For 20 examples: Use r=8-16
  - For 100+ examples: Use r=16-32
  
- **`lora_alpha`** (default: 32): LoRA alpha. Typically 2x `lora_r` for better scaling

- **`lora_dropout`** (default: 0.05): Dropout rate. Higher = more regularization

- **`target_modules`**: Which layers to apply LoRA to
  - Default: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - These are the attention and MLP layers

### Avoiding Catastrophic Forgetting

LoRA helps prevent catastrophic forgetting by:

1. **Only training a small subset of parameters** (~0.1-1% of total parameters)
2. **Keeping original weights frozen** - base model weights remain unchanged
3. **Low-rank adaptation** - learns task-specific patterns without overwriting general knowledge

## Memory Optimization

If you encounter OOM errors:

1. **Reduce batch size:**
   ```bash
   --batch_size 1
   ```

2. **Increase gradient accumulation:**
   ```bash
   --gradient_accumulation_steps 8
   ```

3. **Reduce LoRA rank:**
   ```bash
   --lora_r 8
   ```

4. **Reduce max length:**
   ```bash
   --max_length 1024
   ```

5. **Use gradient checkpointing** (already enabled in script)

## Loading Fine-tuned Model

After training, load the model like this:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load base model
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./qwen2_5_vl_lora_checkpoint")

# Merge LoRA weights (optional, for faster inference)
# model = model.merge_and_unload()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
```

## Integration with Your Agent

Since you're using vLLM server for inference, you have two options:

### Option 1: Use LoRA Adapters with vLLM (Recommended)

Start vLLM server with LoRA support:
```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --port 7999 \
    --trust-remote-code \
    --enable-lora \
    --lora-modules my-lora=./qwen2_5_vl_lora_checkpoint
```

Your existing `starter.py` will automatically use the fine-tuned model - no code changes needed!

### Option 2: Merge LoRA Weights

Merge LoRA weights into base model for simpler deployment:
```bash
python load_finetuned_model.py \
    --lora_path ./qwen2_5_vl_lora_checkpoint \
    --merge \
    --output_path ./qwen2_5_vl_merged
```

Then start vLLM with merged model:
```bash
vllm serve ./qwen2_5_vl_merged \
    --port 7999 \
    --trust-remote-code
```

See `README_VLLM_DEPLOYMENT.md` for detailed instructions.

## Monitoring Training

The script logs:
- Training loss
- Learning rate schedule
- Training steps

Check the output directory for:
- `checkpoint-*`: Training checkpoints
- `adapter_config.json`: LoRA configuration
- `adapter_model.bin`: LoRA weights

## Troubleshooting

### Error: `KeyError: 'qwen2_5_vl'`
- Solution: Install transformers from source:
  ```bash
  pip install git+https://github.com/huggingface/transformers accelerate
  ```

### Out of Memory (OOM)
- Reduce batch size to 1
- Increase gradient accumulation steps
- Reduce LoRA rank
- Use a smaller max_length

### Slow Training
- Use multiple GPUs with `accelerate`
- Increase batch size if memory allows
- Use `flash_attention_2` if available

## Expected Results

With 20 high-quality examples:
- **Training time**: ~30-60 minutes on 1x A100
- **Model size**: ~100-200MB (LoRA weights only)
- **Performance**: Should see improved adherence to single-action format and button/input distinction

## References

- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

