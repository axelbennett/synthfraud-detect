"""
Evaluate the trained forensic classifier against the held-out generator
(e.g. DALL-E, if the model was trained only on Stable Diffusion + real photos).

This is the most important evaluation in the whole project. Accuracy on a
generator seen during training tells you almost nothing about real-world
performance -- the honest question is: does this generalize to a
generator it has never seen? Report this number prominently, including
if it's mediocre. That honesty is the point.

Usage:
    python -m src.models.evaluate --config configs/config.yaml
"""
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import yaml

from src.models.forensic_classifier import DamagePhotoDataset, build_model


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_generalization(config: dict):
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")

    model = build_model(
        backbone=config["model"]["backbone"],
        pretrained=False,
        num_classes=config["model"]["num_classes"],
    )
    model.load_state_dict(torch.load("models_output/forensic_classifier.pt", map_location=device))
    model = model.to(device)
    model.eval()

    held_out = config["data"]["held_out_generator"]
    print(f"Evaluating on held-out generator: {held_out}")

    test_set = DamagePhotoDataset(
        real_dir=config["data"]["raw_dir"],
        synthetic_dir=config["data"]["synthetic_dir"],
        image_size=config["data"]["image_size"],
        only_generators=[held_out],
    )
    test_loader = DataLoader(test_set, batch_size=config["model"]["batch_size"])

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.cpu().tolist())

    print("\n--- Held-out generalization report ---")
    print(classification_report(all_labels, all_preds, target_names=["real", "ai_generated"]))
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC: {auc:.4f}")

    print("\nCompare this to in-distribution validation accuracy from training.")
    print("A large gap here is the honest finding to report, not hide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generalization to held-out generator")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_generalization(config)
