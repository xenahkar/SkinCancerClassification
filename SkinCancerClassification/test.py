import os
import subprocess

import hydra
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from model import ResNet18BinaryClassifier
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from trainer import SkinImageClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    subprocess.run(["dvc", "pull"])

    transformer = transforms.Compose(
        [
            transforms.Resize(
                (config["model"]["image_height"], config["model"]["image_width"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                config["model"]["image_mean"], config["model"]["image_std"]
            ),
        ]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config["data_loading"]["test_data_path"]), transform=transformer
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
    )

    model = ResNet18BinaryClassifier(config["model"]["num_classes"])
    module = SkinImageClassifier.load_from_checkpoint(
        f'{config["model"]["model_local_path"]}/{config["testing"]["checkpoint_name"]}',
        model=model,
        lr=config["training"]["lr"],
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    results = trainer.test(module, dataloaders=test_loader)
    print(results)


if __name__ == "__main__":
    main()
