"""Lesson 1: inspect the raw Qwen3-0.6B checkpoint and relate every weight to the architecture."""
import os
from safetensors.torch import load_file

MODEL_DIR = os.path.expanduser("~/huggingface/Qwen3-0.6B-official")

# Step 1: load everything into a dict {name: tensor}
weights = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
print(f"{len(weights)} tensors in the file\n")

# Step 2: show only the tensors we care about
for name, t in weights.items():
    if "layers.0." in name or "embed_tokens" in name or name == "model.norm.weight":
        print(f"{name:50s} {str(list(t.shape)):20s} {t.dtype}")

# Step 3: count parameters
total = sum(t.numel() for t in weights.values())
embed = weights["model.embed_tokens.weight"].numel()
print(f"\ntotal params:          {total / 1e9:.3f} B  (embedding counted twice: lm_head is a stored copy)")
print(f"unique params:         {(total - embed) / 1e9:.3f} B  (the \"0.6B\" in the name)")
print(f"without any embedding: {(total - 2 * embed) / 1e9:.3f} B  (all 28 layers + final norm)")
num_layers = len({n.split('.')[2] for n in weights if '.layers.' in n})
print(f"layers in the file:    {num_layers}")
print(f"has lm_head.weight?    {'lm_head.weight' in weights}")

# Step 4: largest tensor (embed_tokens and lm_head tie; max() returns the first one it finds)
largest_name, largest_tensor = max(weights.items(), key=lambda x: x[1].numel())
print(f"largest tensor:        {largest_name} {list(largest_tensor.shape)}")


# Step 5: attention ("talk") vs MLP ("think") parameters over all 28 layers
attn_params = 0
mlp_params = 0
for name, t in weights.items():
    if "self_attn" in name:
        attn_params += t.numel()
    elif "mlp" in name:
        mlp_params += t.numel()

print(f"\nself-attention params: {attn_params / 1e6:.3f} M")
print(f"MLP params:            {mlp_params / 1e6:.3f} M")
