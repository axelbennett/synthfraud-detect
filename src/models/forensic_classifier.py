"""
Forensic classifier: predicts P(AI-generated | image) using a CNN backbone
fine-tuned on real (CarDD) vs. synthetic damage photos.

This is the core PyTorch component of the project. Start with a ResNet50
baseline (config.yaml `model.backbone`) before trying anything fancier --
a well-evaluated simple model beats a poorly-evaluated complex one.

Usage:
    python -m src.models.forensic_classifier --config configs/config.yaml
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DamagePhotoDataset(Dataset):
    """
    Binary classification dataset: label 0 = real (CarDD), label 1 = AI-generated.

    Expects:
        real_dir/      -> real CarDD photos
        synthetic_dir/ -> subfolders per generator (e.g. stable_diffusion_xl/, dalle/)

    Pass `exclude_generators` to hold specific generator subfolders out of
    training entirely -- used to build the generalization test set.
    """

    def __init__(self, real_dir: str, synthetic_dir: str, image_size: int = 224,
                 exclude_generators: list = None, only_generators: list = None):
        self.image_size = image_size
        exclude_generators = exclude_generators or []

        self.samples = []  # (path, label)

        for path in Path(real_dir).glob("**/*.jpg"):
            self.samples.append((path, 0))

        for generator_dir in Path(synthetic_dir).iterdir():
            if not generator_dir.is_dir():
                continue
            if generator_dir.name in exclude_generators:
                continue
            if only_generators and generator_dir.name not in only_generators:
                continue
            for path in generator_dir.glob("*.jpg"):
                self.samples.append((path, 1))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label


def build_model(backbone: str = "resnet50", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    if backbone == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def train(config: dict):
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = DamagePhotoDataset(
        real_dir=config["data"]["raw_dir"],
        synthetic_dir=config["data"]["synthetic_dir"],
        image_size=config["data"]["image_size"],
        exclude_generators=[config["data"]["held_out_generator"]],
    )

    n_val = int(len(dataset) * config["data"]["val_split"])
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=config["model"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["model"]["batch_size"])

    model = build_model(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["model"]["epochs"]):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{config['model']['epochs']} "
              f"- train_loss: {total_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f}")

    output_path = Path("models_output") / "forensic_classifier.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


def evaluate(model: nn.Module, data_loader: DataLoader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forensic classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
