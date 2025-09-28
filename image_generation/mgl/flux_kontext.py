import os
import torch
import cv2
import torch.nn as nn
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)


pipe.to("cuda")

rainy_prompts = {
    "1":"change the background into a rainy day, change the sky into grey overcast sky, add visible rain streaks and puddles, ",
    "2":"change the background into a heavy rainy day, change the sky into grey overcast sky, add visible rain streaks and puddles, ",
    "3":"change the background into a rainstorm day, change the sky into very grey overcast sky, add obviously visible rain streaks and puddles, ",
}
# for night, use guidance scale = 5.0
night_prompts = {
  # "1":"change the background into evening, illuminate the scene with streetlights.",
  "2":"change the background into evening, illuminate the scene brighter with streetlights.",
}

foggy_prompts = {
  "1": "turn the background into a foggy day, add visible smog on the street",
  "2": "change the background into a foggy day, add dense fog on the street",
  "3": "change the background into a foggy day, add fog on the street",
}
snow_prompts = {
  "1": "change the background into a snowy day, add visible falling snowflakes"
}
cloudy_prompts = {
  "1": "change the background into a heavy cloudy day",
  "2": "change the background into a heavy overcast day",
  "3": "change the background into a rainy day, change the sky into grey overcast sky, add visible rain streaks and puddles, ",
}
prompts = {
  "rainy": "change the background into a heavy rainy day, change the sky into grey overcast sky, add visible rain streaks and puddles.",
  "night": "change the background into evening, illuminate the scene brighter with streetlights.",
  "foggy": "turn the background into a foggy day, add visible smog on the street",
  "snowy": "change the background into a snowy day, add visible falling snowflakes"
}
# negative_prompt = "low quality, bad quality, sketches, noisy, blurry, jpeg artifacts, deformed, fused objects, poorly drawn features, low definition"

input_images_path = "datasets/MGL/washington120/images/100770066050110_view1.jpg"
input_images_path = "image_generation/mgl/StreetView360.jpg"
# input_images_path = "datasets/MGL/brussels/images/101864422519940_back.jpg"
# input_images_path = "datasets/MGL/tokyo/images/105828182074721_front.jpg"

output_path = "image_generation/mgl"
input_image = load_image(input_images_path)
basename = os.path.splitext(os.path.basename(input_images_path))[0]
################### Args ###################
num_inference_steps = 50
true_cfg_scale = 1.2
guidance_scale = 7.5

image_width, image_height = input_image.size

for i, prompt in prompts.items():
  image = pipe(
    image=input_image,
    prompt=prompt,
    # negative_prompt=negative_prompt,
    true_cfg_scale=true_cfg_scale,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=torch.Generator().manual_seed(42),
  ).images[0]

  resized_image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)
  output_name=f"{basename}_{i}.jpg"
  out_path = os.path.join(output_path, output_name)
  resized_image.save(out_path)