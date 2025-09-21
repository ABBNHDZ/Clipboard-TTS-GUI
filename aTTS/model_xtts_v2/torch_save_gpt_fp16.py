# split_gpt_h.py
import os
import torch
from safetensors.torch import save_file

dir_path = "../model_xtts_v2.1/"

SRC_CKPT = "xttsv2_state_dict.pth" # => "xttsv2_full_fp32.pth"
EXT_FP16 = "_fp16.safetensors"
EXT_BF16 = "_bf16.safetensors"
EXT_E4M3 = "_f8e4.safetensors"
EXT_E5M2 = "_f8e5.safetensors"


print(f"Chargement du checkpoint {SRC_CKPT} ...")
state = torch.load(dir_path + SRC_CKPT, map_location="cpu")
if "state_dict" in state:
    state = state["state_dict"]
elif "model" in state:
    state = state["model"]
save_file(state, dir_path + "xttsv2.safetensors")

FN = "xttsv2"
DST_FP16 = dir_path + FN + EXT_FP16
DST_BF16 = dir_path + FN + EXT_BF16
DST_E4M3 = dir_path + FN + EXT_E4M3
DST_E5M2 = dir_path + FN + EXT_E5M2

print("Conversion vers FP16...")
cur_state = {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(cur_state, DST_FP16)

print("Conversion vers BF16...")
cur_state = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(cur_state, DST_BF16)

# e4m3 (précision > range) → poids
print("Conversion vers F8e4m3...")
cur_state = {k: v.to(torch.float8_e4m3fn) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(cur_state, DST_E4M3)

# e5m2 (range > précision) → activations
print("Conversion vers F8e5m2...")
cur_state = {k: v.to(torch.float8_e5m2) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(cur_state, DST_E5M2)



mel_state = {k: v for k, v in state.items() if (not k.startswith("gpt.") and not k.startswith("hifigan_decoder.") )}
save_file(mel_state, dir_path + "ckpt_mel.safetensors")

cur_state = {k.replace("hifigan_decoder.", ""): v for k, v in state.items() if k.startswith("hifigan_decoder.")}
save_file(cur_state, dir_path + "ckpt_hfd.safetensors")


FN = "ckpt_hfd"
DST_FP16 = dir_path + FN + EXT_FP16
DST_BF16 = dir_path + FN + EXT_BF16
DST_E4M3 = dir_path + FN + EXT_E4M3
DST_E5M2 = dir_path + FN + EXT_E5M2

print("Conversion vers FP16...")
state_fp= {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in cur_state.items()}
save_file(state_fp, DST_FP16)

print("Conversion vers BF16...")
state_fp = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in cur_state.items()}
save_file(state_fp, DST_BF16)

# e4m3 (précision > range) → poids
print("Conversion vers F8e4m3...")
state_fp = {k: v.to(torch.float8_e4m3fn) if torch.is_floating_point(v) else v for k, v in cur_state.items()}
save_file(state_fp, DST_E4M3)

# e5m2 (range > précision) → activations
print("Conversion vers F8e5m2...")
state_fp = {k: v.to(torch.float8_e5m2) if torch.is_floating_point(v) else v for k, v in cur_state.items()}
save_file(state_fp, DST_E5M2)


gpt_state = {k: v for k, v in state.items() if k.startswith("gpt.")}
save_file(gpt_state, dir_path + "ckpt_gpt_gpt.safetensors")


gpt_state = {k.replace(".gpt.", ".gpt1."): v for k, v in state.items() if k.startswith("gpt.")}
gpt_state = {k.replace("gpt.", ""): v for k, v in gpt_state.items() if k.startswith("gpt.")}
state = {k.replace("gpt1.", "gpt."): v for k, v in gpt_state.items()}
save_file(state, dir_path + "ckpt_gpt.safetensors")

FN = "ckpt_gpt"
DST_FP16 = dir_path + FN + EXT_FP16
DST_BF16 = dir_path + FN + EXT_BF16
DST_E4M3 = dir_path + FN + EXT_E4M3
DST_E5M2 = dir_path + FN + EXT_E5M2

print("Conversion vers FP16...")
state_fp= {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_FP16)

print("Conversion vers BF16...")
state_fp = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_BF16)

# e4m3 (précision > range) → poids
print("Conversion vers F8e4m3...")
state_fp = {k: v.to(torch.float8_e4m3fn) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_E4M3)

# e5m2 (range > précision) → activations
print("Conversion vers F8e5m2...")
state_fp = {k: v.to(torch.float8_e5m2) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_E5M2)

gpt_state = {k: v for k, v in state.items() if not k.startswith("gpt.")}
save_file(gpt_state, dir_path + "ckpt_gpt1.safetensors")

FN = "ckpt_gpt1"
DST_FP16 = dir_path + FN + EXT_FP16
DST_BF16 = dir_path + FN + EXT_BF16
DST_E4M3 = dir_path + FN + EXT_E4M3
DST_E5M2 = dir_path + FN + EXT_E5M2

print("Conversion vers FP16...")
state_fp= {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_FP16)

print("Conversion vers BF16...")
state_fp = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_BF16)

# e4m3 (précision > range) → poids
print("Conversion vers F8e4m3...")
state_fp = {k: v.to(torch.float8_e4m3fn) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_E4M3)

# e5m2 (range > précision) → activations
print("Conversion vers F8e5m2...")
state_fp = {k: v.to(torch.float8_e5m2) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_E5M2)


gpt_state = {k: v for k, v in state.items() if k.startswith("gpt.")}
save_file(gpt_state, dir_path + "ckpt_gpt2.safetensors")

FN = "ckpt_gpt2"
DST_FP16 = dir_path + FN + EXT_FP16
DST_BF16 = dir_path + FN + EXT_BF16
DST_E4M3 = dir_path + FN + EXT_E4M3
DST_E5M2 = dir_path + FN + EXT_E5M2

print("Conversion vers FP16...")
state_fp= {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_FP16)

print("Conversion vers BF16...")
state_fp = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_BF16)

# e4m3 (précision > range) → poids
print("Conversion vers F8e4m3...")
state_fp = {k: v.to(torch.float8_e4m3fn) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_E4M3)

# e5m2 (range > précision) → activations
print("Conversion vers F8e5m2...")
state_fp = {k: v.to(torch.float8_e5m2) if torch.is_floating_point(v) else v for k, v in gpt_state.items()}
save_file(state_fp, DST_E5M2)



EXT_FP16 = "_fp16.safetensors"
EXT_BF16 = "_bf16.safetensors"
EXT_E4M3 = "_f8e4.safetensors"
EXT_E5M2 = "_f8e5.safetensors"

dir_path = "../model_xtts_v2.1/"
OUT_DIR = dir_path + "gpt_h/"
os.makedirs(OUT_DIR, exist_ok=True)

# Fonction de quantization
def quantize_tensor(t: torch.Tensor, mode: str):
    if mode == "fp32":
        return t.to(torch.float32)
    elif mode == "fp16":
        return t.to(torch.float16)
    elif mode == "bf16":
        return t.to(torch.bfloat16)
    elif mode == "f8e4":
        return t.to(torch.float8_e4m3fn)
    elif mode == "f8e5":
        return t.to(torch.float8_e5m2)
    else:
        raise ValueError(f"Unsupported dtype {mode}")

# Charger sur CPU uniquement
ckpt = gpt_state

# Vérifier qu'on a bien le bloc gpt_h
keys = list(ckpt.keys())
gpt_h_keys = [k for k in keys if k.startswith("gpt.h.")]
print(f"Nombre de clés dans gpt.h : {len(gpt_h_keys)}")

# Extraire le nombre de couches
layers = set(int(k.split('.')[2]) for k in gpt_h_keys)
num_layers = max(layers) + 1
print(f"Nombre total de couches h : {num_layers}")

LAYERS_PER_SHARD = 3       # découpe par blocs de x couches
# Découper par shard
DTYPE = "fp32"  # "fp16" float16, "bf16" bfloat16, "f8e4" float8_e4m3fn,"f8e4" float8_e5m2
for start in range(0, num_layers, LAYERS_PER_SHARD):
    end = min(start + LAYERS_PER_SHARD, num_layers)
    shard_dict = {}
    for k, v in ckpt.items():
        if k.startswith("gpt.h."):
            layer_id = int(k.split('.')[2])
            if start <= layer_id < end:
                #shard_dict[k] = v
                shard_dict[k] = quantize_tensor(v, DTYPE)

    shard_file = os.path.join(OUT_DIR, f"gpt_h_{start:02d}_{end-1:02d}_" + DTYPE + ".safetensors")
    save_file(shard_dict, shard_file)


LAYERS_PER_SHARD = 5       # découpe par blocs de x couches
# Découper par shard
DTYPE = "fp16"  # "fp16" float16, "bf16" bfloat16, "f8e4" float8_e4m3fn,"f8e4" float8_e5m2
for start in range(0, num_layers, LAYERS_PER_SHARD):
    end = min(start + LAYERS_PER_SHARD, num_layers)
    shard_dict = {}
    for k, v in ckpt.items():
        if k.startswith("gpt.h."):
            layer_id = int(k.split('.')[2])
            if start <= layer_id < end:
                #shard_dict[k] = v
                shard_dict[k] = quantize_tensor(v, DTYPE)

    shard_file = os.path.join(OUT_DIR, f"gpt_h_{start:02d}_{end-1:02d}_" + DTYPE + ".safetensors")
    save_file(shard_dict, shard_file)

# Découper par shard
DTYPE = "bf16"  # "fp16" float16, "bf16" bfloat16, "f8e4" float8_e4m3fn,"f8e4" float8_e5m2
for start in range(0, num_layers, LAYERS_PER_SHARD):
    end = min(start + LAYERS_PER_SHARD, num_layers)
    shard_dict = {}
    for k, v in ckpt.items():
        if k.startswith("gpt.h."):
            layer_id = int(k.split('.')[2])
            if start <= layer_id < end:
                #shard_dict[k] = v
                shard_dict[k] = quantize_tensor(v, DTYPE)

    shard_file = os.path.join(OUT_DIR, f"gpt_h_{start:02d}_{end-1:02d}_" + DTYPE + ".safetensors")
    save_file(shard_dict, shard_file)

LAYERS_PER_SHARD = 6       # découpe par blocs de x couches
# Découper par shard
DTYPE = "f8e4"  # "fp16" float16, "bf16" bfloat16, "f8e4" float8_e4m3fn,"f8e4" float8_e5m2
for start in range(0, num_layers, LAYERS_PER_SHARD):
    end = min(start + LAYERS_PER_SHARD, num_layers)
    shard_dict = {}
    for k, v in ckpt.items():
        if k.startswith("gpt.h."):
            layer_id = int(k.split('.')[2])
            if start <= layer_id < end:
                #shard_dict[k] = v
                shard_dict[k] = quantize_tensor(v, DTYPE)

    shard_file = os.path.join(OUT_DIR, f"gpt_h_{start:02d}_{end-1:02d}_" + DTYPE + ".safetensors")
    save_file(shard_dict, shard_file)

# Découper par shard
DTYPE = "f8e5"  # "fp16" float16, "bf16" bfloat16, "f8e4" float8_e4m3fn,"f8e5" float8_e5m2
for start in range(0, num_layers, LAYERS_PER_SHARD):
    end = min(start + LAYERS_PER_SHARD, num_layers)
    shard_dict = {}
    for k, v in ckpt.items():
        if k.startswith("gpt.h."):
            layer_id = int(k.split('.')[2])
            if start <= layer_id < end:
                #shard_dict[k] = v
                shard_dict[k] = quantize_tensor(v, DTYPE)

    shard_file = os.path.join(OUT_DIR, f"gpt_h_{start:02d}_{end-1:02d}_" + DTYPE + ".safetensors")
    save_file(shard_dict, shard_file)




# Sauvegarder aussi les autres parties hors gpt.h
other_dict = {k: v for k, v in ckpt.items() if not k.startswith("gpt.h.")}
save_file(other_dict, os.path.join(dir_path, "gpt_ln.safetensors"))

#other_dict = {k: quantize_tensor(v, DTYPE) for k, v in ckpt.items() if not k.startswith("gpt.h.")}
#save_file(other_dict, os.path.join(OUT_DIR, "gpt_ln.sftf16"))
print("Sauvegardé gpt_ln.sft")

FN = "gpt_ln"
DST_FP16 = dir_path + FN + EXT_FP16
DST_BF16 = dir_path + FN + EXT_BF16
DST_E4M3 = dir_path + FN + EXT_E4M3
DST_E5M2 = dir_path + FN + EXT_E5M2


state = other_dict

print("Conversion vers FP16...")
state_fp= {k: v.to(torch.float16) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_FP16)

print("Conversion vers BF16...")
state_fp = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_BF16)

# e4m3 (précision > range) → poids
print("Conversion vers F8e4m3...")
state_fp = {k: v.to(torch.float8_e4m3fn) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_E4M3)

# e5m2 (range > précision) → activations
print("Conversion vers F8e5m2...")
state_fp = {k: v.to(torch.float8_e5m2) if torch.is_floating_point(v) else v for k, v in state.items()}
save_file(state_fp, DST_E5M2)

