#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build KITTI dataset night weather perturbations (Simplified Version)
"""
import torch
import os
import glob
import shutil
import numpy as np
from tqdm import tqdm
from diffusers.utils import load_image
from diffusers import StableDiffusionXLPipeline, EDMEulerScheduler, StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL
from PIL import Image, UnidentifiedImageError

# --- Fixed Parameters ---
# Source and destination paths
src_dir = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/datasets/kitti" 
dest_dir = "/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/datasets/weather_noise"

# Model paths
edit_file_local_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/models/cosxl/cosxl_edit.safetensors"
vae_local_path = "/hkfs/work/workspace/scratch/tj3409-rdaionet/models/sdxl-vae-fp16-fix"

# Generation parameters
seed = 18457
guidance_scale = 6.75
image_scale = 2.5
steps = 100

# --- Load Model ---
print(f"Loading diffusion model...")
vae = AutoencoderKL.from_pretrained(vae_local_path, torch_dtype=torch.float16)
pipe_edit = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    edit_file_local_path, num_in_channels=8, is_cosxl_edit=True, vae=vae, torch_dtype=torch.float16,
)
pipe_edit.scheduler = EDMEulerScheduler(sigma_min=0.002, sigma_max=120.0, sigma_data=1.0, prediction_type="v_prediction", sigma_schedule="exponential")
pipe_edit.to("cuda")
pipe_edit.set_progress_bar_config(disable=True)
pipe_edit.enable_model_cpu_offload()
generator = torch.Generator("cuda")
print(f"Model successfully loaded to CUDA device: {torch.cuda.current_device()}")

# --- Image Resizing Function ---
def resize_image(image):
    """Resizes a PIL image so its width and height are multiples of 8."""
    original_width, original_height = image.size
    new_width = original_width - original_width % 8
    new_height = original_height - original_height % 8
    if new_width <= 0 or new_height <= 0:
        print(f"Warning: Original image size ({original_width}x{original_height}) results in non-positive dimensions after rounding down to multiple of 8 ({new_width}x{new_height}). Skipping resize.")
        return image
    resized_img = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_img

# --- Prompts ---
night_prompt = "Transform it into deep dark night. Make the sky and street dark. Create stark, high-contrast lighting effect."
negative_prompt = "low quality, bad quality, worst quality, sketches, noisy, blurry, jpeg artifacts, deformed, fused objects, poorly drawn features, low definition"

# --- Image Generation Function ---
def create_variant(image):
    """Generates a night variant using the global pipe_edit."""
    current_generator = torch.Generator("cuda")
    if seed != 0:
        current_generator.manual_seed(seed)
    else:
        current_generator.seed()

    new_width, new_height = image.size
    edited_image = pipe_edit(
            prompt=night_prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=new_height,
            width=new_width,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_scale,
            num_inference_steps=steps,
            generator=current_generator,
        ).images[0]
    return edited_image

# --- Copy Directory Structure and Non-Image Files ---
def copy_directory_structure(src_folder, dest_folder):
    """Copy directory structure from specific directory"""
    print(f"Copying directory structure: {src_folder} -> {dest_folder}")
    for dirpath, dirnames, _ in os.walk(src_folder):
        structure = os.path.relpath(dirpath, src_folder)
        if structure == ".":  # Skip root directory
            continue
        new_dir = os.path.join(dest_folder, structure)
        os.makedirs(new_dir, exist_ok=True)

def copy_non_image_files(src_folder, dest_folder):
    """Copy all non-image files"""
    non_image_files = []
    print(f"Finding non-image files: {src_folder}")
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in filenames:
            if not filename.lower().endswith(('.png')):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, src_folder)
                non_image_files.append(rel_path)

    print(f"Found {len(non_image_files)} non-image files. Copying...")
    for file_path in tqdm(non_image_files, desc=f"Copying non-image files"):
        src_file_path = os.path.join(src_folder, file_path)
        dest_file_path = os.path.join(dest_folder, file_path)
        os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
        try:
            shutil.copy2(src_file_path, dest_file_path)
        except Exception as e:
            print(f"Warning: Could not copy {src_file_path} to {dest_file_path}: {e}")

# --- Image Processing Function ---
def process_image(image_path, src_folder, dest_folder):
    try:
        image = load_image(image_path)
        image_resized = resize_image(image)

        if image_resized.size[0] <= 0 or image_resized.size[1] <= 0:
            print(f"Warning: Image {image_path} has invalid dimensions after resizing. Skipping.")
            return

        rel_path = os.path.relpath(image_path, src_folder)
        
        night_image = create_variant(image=image_resized)

        target_path = os.path.join(dest_folder, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        night_image.save(target_path)

    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}. Skipping.")
    except UnidentifiedImageError:
        print(f"Warning: Could not identify or open image file: {image_path}. Skipping.")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# --- Main Function ---
def main():
    if not torch.cuda.is_available():
        print(f"FATAL ERROR: CUDA is not available. Exiting.")
        exit(1)
    
    # Create destination root directory
    night_dest_dir = os.path.join(dest_dir, "night")
    os.makedirs(night_dest_dir, exist_ok=True)
    
    # Process specific two directories
    target_folders = ["2011_09_26"]
    
    for folder in target_folders:
        src_folder = os.path.join(src_dir, folder)
        dest_folder = os.path.join(night_dest_dir, folder)
        
        if not os.path.exists(src_folder):
            print(f"Warning: Source directory does not exist: {src_folder}. Skipping.")
            continue
        
        # Create destination folder
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy directory structure and non-image files
        copy_directory_structure(src_folder, dest_folder)
        copy_non_image_files(src_folder, dest_folder)
        
        # Find and process images
        print(f"Finding images in directory {src_folder}...")
        image_files = []
        patterns = ["**/image_[0][2]/data/*.png"]
        
        for pattern in patterns:
            found = glob.glob(os.path.join(src_folder, pattern), recursive=True)
            if found:
                image_files.extend(found)
        
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            print(f"Error: No image files found in {src_folder}.")
            continue
        
        print(f"Found {len(image_files)} images. Starting processing...")
        
        # Process all images
        for img_path in tqdm(image_files, desc=f"Processing images in {folder}"):
            process_image(img_path, src_folder, dest_folder)
        
        print(f"{folder} directory processing complete!")
        
        # Create marker file
        marker_path = os.path.join(dest_folder, ".generated_night_v1")
        try:
            with open(marker_path, "w") as f:
                device_name = torch.cuda.get_device_name(torch.cuda.current_device())
                f.write(f"This directory contains night variants generated on {device_name}.")
        except Exception as e:
            print(f"Warning: Could not write marker file {marker_path}: {e}")
    
    print("night variant dataset build complete!")

if __name__ == "__main__":
    main()
