# Using LoRA Adapters with AWQ Models for Efficient Inference

## Overview

**Yes, you can fine-tune on the base model and use LoRA adapters with the AWQ model for inference!**

This workflow gives you:
- ✅ **Cheaper training**: Fine-tune on base model (no quantization issues)
- ✅ **Cheaper inference**: Use AWQ quantized model (4x smaller, faster)
- ✅ **Best of both worlds**: Combine fine-tuned adapters with efficient inference

## How It Works

LoRA adapters are **architecture-agnostic** - they modify the same layers regardless of whether the base model is quantized or not. When you train LoRA on a base model, the adapter weights can be applied to the AWQ version of the same model for inference.

## Workflow

### Step 1: Fine-tune on Base Model

```bash
# Train LoRA adapters on base model (no AWQ issues)
./run_qwen32b_awq.sh
```

This creates LoRA adapters in `./qwen2_5_vl_lora_checkpoint/`

### Step 2: Verify LoRA Adapter with AWQ Model

Test that your LoRA adapter works with the AWQ model:

```bash
python load_finetuned_model.py \
    --lora_path ./qwen2_5_vl_lora_checkpoint \
    --base_model Qwen/Qwen2.5-VL-32B-Instruct-AWQ
```

This will verify compatibility and show you the vLLM command.

### Step 3: Deploy with vLLM (AWQ + LoRA)

Start vLLM server with AWQ model and your LoRA adapters:

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --port 7999 \
    --trust-remote-code \
    --enable-lora \
    --lora-modules my-lora=./qwen2_5_vl_lora_checkpoint
```

**Benefits:**
- Uses AWQ quantization (4x memory reduction, faster inference)
- Applies your fine-tuned LoRA adapters
- Same API as before - no code changes needed!

## Important Notes

### ✅ What Works

1. **LoRA adapters trained on base model → AWQ model**: ✅ Works perfectly
2. **vLLM with LoRA adapters on AWQ models**: ✅ Fully supported
3. **Dynamic adapter switching**: ✅ Can load multiple adapters

### ⚠️ What Doesn't Work

1. **Merging LoRA into AWQ models**: ❌ Not recommended
   - Merging should be done with base models
   - If you need a merged model, merge into base first, then quantize

2. **Training on AWQ models**: ❌ Not supported
   - AutoAWQ is deprecated and incompatible
   - Always train on base models

## Comparison: Base vs AWQ Inference

| Aspect | Base Model | AWQ Model |
|--------|-----------|-----------|
| **Memory** | ~64GB | ~16GB (4x smaller) |
| **Speed** | Baseline | ~2-3x faster |
| **Cost** | Higher | Lower |
| **LoRA Support** | ✅ Yes | ✅ Yes (via vLLM) |
| **Training** | ✅ Yes | ❌ No |

## Alternative: Merged Model (If Needed)

If you need a standalone merged model (without separate adapters):

```bash
# Step 1: Merge LoRA into BASE model
python load_finetuned_model.py \
    --lora_path ./qwen2_5_vl_lora_checkpoint \
    --base_model Qwen/Qwen2.5-VL-32B-Instruct \
    --merge \
    --output_path ./qwen2_5_vl_merged

# Step 2: Use merged model with vLLM
vllm serve ./qwen2_5_vl_merged \
    --port 7999 \
    --trust-remote-code
```

**Note**: Merged model will be full precision (not AWQ). For AWQ benefits, use LoRA adapters directly instead.

## Recommended Approach

**For production inference with cost efficiency:**

1. ✅ Fine-tune on base model: `./run_qwen32b_awq.sh`
2. ✅ Deploy AWQ model with LoRA adapters via vLLM
3. ✅ Get best of both worlds: fine-tuning + efficient inference

This gives you the cheapest inference while maintaining your fine-tuned behavior!

## Troubleshooting

### vLLM doesn't load LoRA adapter

- Ensure vLLM version supports LoRA: `pip install --upgrade vllm`
- Check adapter path is correct
- Verify adapter was trained on compatible model architecture

### Adapter not applying fine-tuned behavior

- Verify adapter loaded: Check vLLM logs for "Loaded LoRA adapter"
- Test with simple prompt to confirm behavior changed
- Ensure you're using the correct adapter name in requests

### Memory issues

- AWQ model should use ~16GB instead of ~64GB
- If still OOM, reduce `--max-model-len` in vLLM
- Use `--enforce-eager` for lower memory usage

## Summary

**Your workflow is correct!** Fine-tune on base model, then use LoRA adapters with AWQ model for inference. This is the recommended approach for cost-effective production deployment.

