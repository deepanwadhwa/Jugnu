#!/usr/bin/env python3
import os
import sys
import time
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
import safetensors.torch
from huggingface_hub import hf_hub_download
import psutil

REPO = "Qwen/Qwen3.6-35B-A3B"

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_reference_vision_model():
    print("Fetching index...")
    index_path = hf_hub_download(repo_id=REPO, filename="model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)
    weight_map = index_data["weight_map"]
    
    visual_shards = sorted(list(set(v for k, v in weight_map.items() if "visual" in k)))
    print(f"Visual shards: {visual_shards}")
    
    ref_weights = {}
    for shard in visual_shards:
        local_path = hf_hub_download(repo_id=REPO, filename=shard)
        with safetensors.torch.safe_open(local_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "visual" in k:
                    ref_weights[k.replace("model.visual.", "")] = f.get_tensor(k)
                    
    print("Loading config...")
    config = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    
    print("Instantiating model...")
    model = Qwen3_5MoeVisionModel(config.vision_config)
    model.eval()
    model.load_state_dict(ref_weights)
    return model

def main():
    os.makedirs("tests/fixtures/vision", exist_ok=True)
    os.makedirs("docs/regressions/vision-cost", exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    
    print(f"Loading reference model (BF16)... Initial RSS: {get_memory_mb():.1f} MB")
    model = load_reference_vision_model()
    # model = model.to(torch.bfloat16) # we will just run it in FP32/BF16 to get valid outputs
    print(f"Model loaded. RSS: {get_memory_mb():.1f} MB")

    # --- E-V2: Fixtures ---
    print("\n--- E-V2: Generating Fixtures ---")
    fixture_images = [
        Image.new("RGB", (256, 256), color="red"),
        Image.new("RGB", (512, 256), color="blue"),
        Image.new("RGB", (256, 512), color="green")
    ]
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            val = output[0] if isinstance(output, tuple) else output
            activations[name] = val.detach().cpu().to(torch.float32).numpy()
        return hook

    handles = []
    handles.append(model.patch_embed.register_forward_hook(get_activation("post_patch_embed")))
    handles.append(model.blocks[0].register_forward_hook(get_activation("post_block_0")))
    handles.append(model.blocks[13].register_forward_hook(get_activation("post_block_13")))
    handles.append(model.blocks[26].register_forward_hook(get_activation("post_block_26")))

    for i, img in enumerate(fixture_images):
        print(f"Processing fixture image {i} ({img.size})...")
        inputs = processor(images=img, text="Describe", return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["image_grid_thw"]
        
        with torch.no_grad():
            merger_out = model(pixel_values, grid_thw)[0]
            activations["post_merger"] = merger_out.detach().cpu().to(torch.float32).numpy()
            
        for stage, data in activations.items():
            path = f"tests/fixtures/vision/img{i}_{stage}.npy"
            np.save(path, data)
            print(f"  Saved {stage} to {path}")
            
    for h in handles:
        h.remove()

    # --- E-V5: Cost Measurement ---
    print("\n--- E-V5: Measuring Cost ---")
    resolutions = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
    threads_to_test = [2, 4]
    
    results = []
    
    for threads in threads_to_test:
        torch.set_num_threads(threads)
        print(f"\nTesting with {threads} threads:")
        
        for res in resolutions:
            img = Image.new("RGB", res, color="white")
            inputs = processor(images=img, text="Describe", return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            grid_thw = inputs["image_grid_thw"]
            
            # Warmup
            with torch.no_grad():
                _ = model(pixel_values, grid_thw)
                
            # Measure
            start_rss = get_memory_mb()
            start_time = time.time()
            
            with torch.no_grad():
                out = model(pixel_values, grid_thw)[0]
                
            end_time = time.time()
            end_rss = get_memory_mb()
            
            dur = end_time - start_time
            lm_tokens = out.shape[1]
            
            print(f"  {res[0]}x{res[1]}: {lm_tokens} tokens, {dur:.2f}s, Peak RSS Delta: {end_rss - start_rss:.1f} MB")
            
            results.append({
                "resolution": f"{res[0]}x{res[1]}",
                "threads": threads,
                "lm_tokens": lm_tokens,
                "vit_seconds": dur,
                "rss_delta_mb": end_rss - start_rss
            })
            
    with open("docs/regressions/vision-cost/report.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
