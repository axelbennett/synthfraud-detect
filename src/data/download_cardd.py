"""
Download the CarDD real car-damage-photo dataset.

CarDD: 4,000 high-resolution car damage images, 9,000+ annotated instances
across 6 damage categories (dent, scratch, crack, glass_shatter, lamp_broken,
tire_flat). Used here as the "real" class for the forensic classifier.

Usage:
    python -m src.data.download_cardd
"""
import argparse
import yaml
from pathlib import Path

from datasets import load_dataset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_cardd(hf_dataset_name: str, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{hf_dataset_name}' from Hugging Face...")
    dataset = load_dataset(hf_dataset_name)

    for split_name, split_data in dataset.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving {len(split_data)} images from split '{split_name}'...")
        for idx, example in enumerate(split_data):
            image = example["image"]
            image_path = split_dir / f"cardd_{split_name}_{idx:05d}.jpg"
            image.save(image_path)

        print(f"  -> saved to {split_dir}")

    print(f"Done. CarDD dataset available at: {output_path}")
    print("NOTE: images are labeled 'real' for this project's purposes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CarDD dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    download_cardd(
        hf_dataset_name=config["data"]["huggingface_dataset"],
        output_dir=config["data"]["raw_dir"],
    )
