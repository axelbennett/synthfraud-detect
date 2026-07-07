"""
Generate the synthetic (AI-generated) car damage photo set used to train and
evaluate the forensic classifier.

Two generator families are used deliberately:
  1. Stable Diffusion / SDXL (open-source, run locally)
  2. DALL-E via API (closed-source, different generation fingerprint)

Training on multiple generator families and holding one out entirely for
testing is the single most important design decision in this pipeline --
it's what lets you honestly measure generalization instead of overfitting
to one generator's artifacts. See config.yaml `held_out_generator`.

A fraction of the synthetic set is produced via img2img editing of real
CarDD photos rather than pure text-to-image generation, since partially
AI-edited real photos are a harder and more realistic fraud pattern than
fully synthetic images.

Usage:
    python -m src.data.generate_synthetic --generator stable_diffusion_xl
    python -m src.data.generate_synthetic --generator dalle
"""
import argparse
import random
from pathlib import Path

import yaml


DAMAGE_PROMPTS = {
    "dent": "close-up photo of a car door with a deep dent, realistic damage, daylight, insurance claim photo",
    "scratch": "close-up photo of a car bumper with a long scratch exposing paint layers, realistic, daylight",
    "crack": "close-up photo of a cracked car windshield, spiderweb crack pattern, realistic damage photo",
    "glass_shatter": "close-up photo of a shattered car side window, glass fragments, realistic damage photo",
    "lamp_broken": "close-up photo of a broken car headlight, cracked plastic lens, realistic damage photo",
    "tire_flat": "close-up photo of a flat car tire on pavement, realistic, daylight, insurance claim photo",
}


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_with_sdxl(n_images: int, categories: list, output_dir: Path) -> None:
    """Generate images locally using Stable Diffusion XL via diffusers."""
    from diffusers import StableDiffusionXLPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        category = random.choice(categories)
        prompt = DAMAGE_PROMPTS[category]
        image = pipe(prompt, num_inference_steps=30).images[0]
        image.save(output_dir / f"sdxl_{category}_{i:05d}.jpg")
        if i % 50 == 0:
            print(f"  generated {i}/{n_images}")


def generate_with_dalle(n_images: int, categories: list, output_dir: Path) -> None:
    """Generate images via OpenAI's DALL-E API (second generator family,
    used as the held-out generalization test set by default)."""
    import requests
    from openai import OpenAI

    client = OpenAI()  # expects OPENAI_API_KEY in environment
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_images):
        category = random.choice(categories)
        prompt = DAMAGE_PROMPTS[category]
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
        )
        image_url = response.data[0].url
        image_bytes = requests.get(image_url).content
        with open(output_dir / f"dalle_{category}_{i:05d}.jpg", "wb") as f:
            f.write(image_bytes)
        if i % 50 == 0:
            print(f"  generated {i}/{n_images}")


def generate_img2img_edits(real_image_dir: Path, n_edits: int, output_dir: Path) -> None:
    """
    Take real CarDD photos and apply AI inpainting/img2img to alter or
    exaggerate damage -- simulates the 'real photo, AI-touched-up' fraud
    pattern, which is harder to detect than fully synthetic images.

    Stub: fill in with diffusers img2img/inpainting pipeline once the
    real dataset is downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    real_images = list(real_image_dir.glob("**/*.jpg"))[:n_edits]
    print(f"TODO: apply img2img editing to {len(real_images)} real photos")
    print(f"Output would be saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic damage photos")
    parser.add_argument(
        "--generator",
        choices=["stable_diffusion_xl", "dalle", "img2img_edits"],
        required=True,
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    categories = config["synthetic_generation"]["damage_categories"]
    output_base = Path(config["data"]["synthetic_dir"])

    if args.generator == "stable_diffusion_xl":
        gen_cfg = next(
            g for g in config["synthetic_generation"]["generators"]
            if g["name"] == "stable_diffusion_xl"
        )
        generate_with_sdxl(gen_cfg["n_images"], categories, output_base / "stable_diffusion_xl")

    elif args.generator == "dalle":
        gen_cfg = next(
            g for g in config["synthetic_generation"]["generators"]
            if g["name"] == "dalle"
        )
        generate_with_dalle(gen_cfg["n_images"], categories, output_base / "dalle")

    elif args.generator == "img2img_edits":
        real_dir = Path(config["data"]["raw_dir"])
        n_edits = int(
            config["synthetic_generation"]["img2img_edit_fraction"]
            * sum(g["n_images"] for g in config["synthetic_generation"]["generators"])
        )
        generate_img2img_edits(real_dir, n_edits, output_base / "img2img_edits")
