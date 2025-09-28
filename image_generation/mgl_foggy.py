#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build MGL dataset with foggy perturbations using distributed processing across multiple nodes.
Each node processes a subset of images using its local GPUs.
"""

import torch
import time
import torch.multiprocessing as mp
import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image, UnidentifiedImageError
from typing import List

# --- Fixed Parameters ---
src_dir = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/datasets/MGL"
dest_dir = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/datasets/MGL_foggy"

PROMPT_CONFIG = {
    "prompt": "turn the background into a foggy day, add visible smog on the street",
    "seed": 42,
    "num_inference_steps": 50,
    "true_cfg_scale": 1.2,
    "guidance_scale": 7.5,
}

# --- Utility Functions (unchanged) ---
def copy_directory_structure(src_folder, dest_folder):
    """Copy directory structure from source to destination."""
    print(f"Copying directory structure: {src_folder} -> {dest_folder}")
    for dirpath, _, _ in os.walk(src_folder):
        structure = os.path.relpath(dirpath, src_folder)
        if structure == ".":
            continue
        new_dir = os.path.join(dest_folder, structure)
        os.makedirs(new_dir, exist_ok=True)

def create_variant(pipe, input_image, device, prompt_config):
    """Generates a foggy variant of a single image using the provided pipeline."""
    image_width, image_height = input_image.size
    generator = torch.Generator(device=device).manual_seed(prompt_config["seed"])

    image = pipe(
        image=input_image,
        prompt=prompt_config["prompt"],
        true_cfg_scale=prompt_config["true_cfg_scale"],
        num_inference_steps=prompt_config["num_inference_steps"],
        guidance_scale=prompt_config["guidance_scale"],
        generator=generator,
    ).images[0]
    
    resized_image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
    return resized_image

def process_chunk(local_gpu_rank: int, local_world_size: int, node_image_files: List[str], 
                  src_root: str, dest_root: str, prompt_config: dict, node_rank: int):
    """
    Process images on a single GPU within a node.
    
    Args:
        local_gpu_rank: GPU index within the current node (0-3)
        local_world_size: Number of GPUs in current node (4)
        node_image_files: Images assigned to current node
        src_root: Source directory
        dest_root: Destination directory  
        prompt_config: Generation parameters
        node_rank: Current node rank
    """
    device = f"cuda:{local_gpu_rank}"
    print(f"[Node {node_rank}, GPU {local_gpu_rank}] Initializing on device: {device}")

    # Load model on current GPU
    try:
        pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        )
        pipe.to(device)
        print(f"[Node {node_rank}, GPU {local_gpu_rank}] Model loaded successfully.")
    except Exception as e:
        print(f"[Node {node_rank}, GPU {local_gpu_rank}] FATAL: Failed to load model. Error: {e}")
        return

    # Divide node's images among local GPUs
    total_node_images = len(node_image_files)
    images_per_gpu = total_node_images // local_world_size
    start_idx = local_gpu_rank * images_per_gpu
    end_idx = (local_gpu_rank + 1) * images_per_gpu if local_gpu_rank != local_world_size - 1 else total_node_images
    
    gpu_image_chunk = node_image_files[start_idx:end_idx]
    
    print(f"[Node {node_rank}, GPU {local_gpu_rank}] Processing {len(gpu_image_chunk)} images (indices {start_idx}-{end_idx})")

    # Process images
    for image_path in tqdm(gpu_image_chunk, desc=f"Node{node_rank}-GPU{local_gpu_rank}", position=node_rank*4+local_gpu_rank):
        try:
            rel_path = os.path.relpath(image_path, src_root)
            target_path = os.path.join(dest_root, rel_path)
            
            if os.path.exists(target_path):
                continue
            
            input_image = load_image(image_path)
            foggy_image = create_variant(pipe, input_image, device, prompt_config)

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            foggy_image.save(target_path)

        except Exception as e:
            print(f"[Node {node_rank}, GPU {local_gpu_rank}] Error processing {image_path}: {e}")

def main():
    """Main function with distributed processing support."""
    parser = argparse.ArgumentParser(description="Distributed MGL foggy variant generation.")
    parser.add_argument("--scene", required=True, 
                       choices=["antwerp", "berlin", "brussels", "hoboken", "los_angeles", 
                              "milan", "paris", "sanfrancisco", "tokyo", "washington"])
    parser.add_argument("--node-rank", type=int, required=True, help="Current node rank (0-based)")
    parser.add_argument("--total-nodes", type=int, required=True, help="Total number of nodes")
    
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FATAL ERROR: CUDA not available.")
        return

    local_world_size = torch.cuda.device_count()  # Should be 4 per node
    print(f"[Node {args.node_rank}] Found {local_world_size} local GPUs")

    # Setup paths
    src_folder = os.path.join(src_dir, args.scene)
    dest_folder = os.path.join(dest_dir, args.scene)
    
    if not os.path.exists(src_folder):
        print(f"ERROR: Source directory not found: {src_folder}")
        return

    # Only node 0 handles directory setup to avoid conflicts
    if args.node_rank == 0:
        print(f"[Node 0] Setting up directory structure for {args.scene}")
        os.makedirs(dest_folder, exist_ok=True)
        copy_directory_structure(src_folder, dest_folder)
    
    # Wait for node 0 to finish setup (simple barrier)
    time.sleep(10 * args.node_rank)  # Stagger node startup

    # Find all images
    print(f"[Node {args.node_rank}] Finding images...")
    image_search_path = os.path.join(src_folder, 'images', '*.jpg')
    all_images = sorted(glob.glob(image_search_path, recursive=True))
    
    if not all_images:
        print(f"[Node {args.node_rank}] No images found!")
        return

    # Distribute images across nodes
    total_images = len(all_images)
    images_per_node = total_images // args.total_nodes
    node_start = args.node_rank * images_per_node
    node_end = (args.node_rank + 1) * images_per_node if args.node_rank != args.total_nodes - 1 else total_images
    
    node_images = all_images[node_start:node_end]
    
    print(f"[Node {args.node_rank}] Assigned {len(node_images)} images (total: {total_images})")
    print(f"[Node {args.node_rank}] Starting local GPU processes...")

    # Spawn local GPU processes
    mp.spawn(
        process_chunk,
        args=(local_world_size, node_images, src_folder, dest_folder, PROMPT_CONFIG, args.node_rank),
        nprocs=local_world_size,
        join=True
    )

    print(f"[Node {args.node_rank}] Completed processing {len(node_images)} images")

    # Node 0 creates completion marker
    if args.node_rank == 0:
        marker_path = os.path.join(dest_folder, ".generated_foggy_distributed")
        try:
            with open(marker_path, "w") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Distributed foggy variants generated at {timestamp}\n")
                f.write(f"Total nodes: {args.total_nodes}\n")
                f.write(f"Total images: {total_images}\n")
        except Exception as e:
            print(f"Warning: Could not write marker file: {e}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()