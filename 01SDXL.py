# --*-- coding:utf-8 --*--
# @Author : 一只楚楚猫
# @File : 01SDXL.py
# @Software : PyCharm

from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch

unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = "a close-up picture of an old man standing in the rain"
image = pipe(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]