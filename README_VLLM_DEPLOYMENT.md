# Deploying Fine-Tuned Model with vLLM

After fine-tuning your Qwen2.5-VL model with LoRA, you have two options for deploying with vLLM:

## Option 1: Use LoRA Adapters Directly (Recommended)

**Pros:**
- Keep adapters separate (easier to update/swap)
- Lower disk usage
- Can load multiple adapters

**Cons:**
- Requires specifying adapter path
- Slightly more complex setup

### Steps:

1. **Start vLLM server with LoRA support:**

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --port 7999 \
    --trust-remote-code \
    --enable-lora \
    --lora-modules my-lora=./qwen2_5_vl_lora_checkpoint
```

2. **Use the model:**

The model will be available at `http://localhost:7999` with the same API as before. The LoRA adapter will be automatically applied.

## Option 2: Merge LoRA Weights (Simpler)

**Pros:**
- Single model file (simpler deployment)
- Slightly faster inference
- No need to specify adapter path

**Cons:**
- Harder to update later (need to retrain and merge)
- Larger disk usage (full model size)

### Steps:

1. **Merge LoRA weights into base model:**

```bash
python load_finetuned_model.py \
    --lora_path ./qwen2_5_vl_lora_checkpoint \
    --merge \
    --output_path ./qwen2_5_vl_merged
```

2. **Start vLLM server with merged model:**

```bash
vllm serve ./qwen2_5_vl_merged \
    --port 7999 \
    --trust-remote-code
```

3. **Use the model:**

Same as before - the merged model will have your fine-tuned weights baked in.

## Verification

To verify your LoRA adapter loads correctly (without merging):

```bash
python load_finetuned_model.py \
    --lora_path ./qwen2_5_vl_lora_checkpoint
```

## Integration with Your Agent

Your `starter.py` already uses the vLLM server at `localhost:7999`, so after deploying:

1. **If using LoRA adapters directly:** The server will automatically apply the adapter
2. **If using merged model:** Just point vLLM to the merged model path

No changes needed to your agent code - it will automatically use the fine-tuned model!

## Troubleshooting

### vLLM doesn't recognize LoRA adapter

Make sure you're using a recent version of vLLM:
```bash
pip install --upgrade vllm
```

### Out of memory errors

- Use the AWQ quantized model (already using it)
- Reduce `--max-model-len` if needed
- Use `--enforce-eager` for lower memory usage

### Model not applying fine-tuned behavior

- Verify LoRA adapter loaded: Check vLLM logs for "Loaded LoRA adapter"
- Test with a simple prompt to see if behavior changed
- If merged, verify merge completed successfully

## Performance Notes

- **LoRA adapters:** Minimal overhead (~1-2% slower)
- **Merged model:** No overhead, same speed as base model
- **Memory:** Both use similar memory (LoRA adds ~100-200MB)

## Recommendation

For production: **Use Option 1 (LoRA adapters directly)** - easier to update and experiment with different adapters.

For simplicity: **Use Option 2 (merge weights)** - single model file, simpler deployment.

