import json
import os

import safetensors.torch
import torch

# This is somewhat hacky, and I should probably feel bad...but I don't.
to_replace = {
    ".": "_",
    "_lora_up_": ".lora_up.",
    "_lora_down_": ".lora_down.",
    "unet_": "lora_unet_",
}

# Check for missing alpha keys, set to 0.8 if not found
base_attn_keys = [
    "attn1_to_out",
    "attn1_to_k",
    "attn1_to_q",
    "attn1_to_v",
    "attn2_to_out",
    "attn2_to_k",
    "attn2_to_q",
    "attn2_to_v",
    "self_attn_k_proj",
    "self_attn_q_proj",
    "self_attn_v_proj",
    "self_attn_out_proj"
]

secondary_keys = [
    "lora_down.weight",
    "lora_up.weight"
]


def convert_diffusers_to_kohya_lora(path, metadata, alpha=0.8):
    model_dict = safetensors.torch.load_file(path)
    new_model_dict = {}
    alpha_keys = []
    # Replace the things
    for (key, v) in model_dict.items():
        for (kc,vc) in to_replace.items():
            key = key.replace(kc, vc)            
        akey = key

        # Check for missing alpha keys
        for k in base_attn_keys:
            if k in akey:
                for rep in secondary_keys:
                    akey = akey.replace(rep, "alpha")
                if akey not in alpha_keys:
                    alpha_keys.append(akey)
        new_model_dict[key] = v

    # Add missing alpha keys
    for k in alpha_keys:
        if k not in new_model_dict:
            new_model_dict[k] = torch.tensor(alpha)
    conv_path = path.replace(".safetensors", "_auto.safetensors")
    safetensors.torch.save_file(new_model_dict, conv_path, metadata=metadata)
    # Delete the file at path, move the new file to path
    os.remove(path)
    os.rename(conv_path, path)
