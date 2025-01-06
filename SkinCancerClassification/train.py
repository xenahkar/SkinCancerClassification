import os
import subprocess

import hydra
import mlflow
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from model import ResNet18BinaryClassifier
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from trainer import SkinImageClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
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

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config["data_loading"]["train_data_path"]), transform=transformer
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(config["data_loading"]["val_data_path"]), transform=transformer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = ResNet18BinaryClassifier(config["model"]["num_classes"])
    module = SkinImageClassifier(model, lr=config["training"]["lr"])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=config["model"]["model_local_path"],
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = pl.loggers.MLFlowLogger(
        experiment_name=config["logging"]["experiment_name"],
        run_name=config["logging"]["run_name"],
        save_dir=config["logging"]["mlflow_save_dir"],
        tracking_uri=config["logging"]["tracking_uri"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_artifact("train.py")
        mlflow.log_param("batch_size", config["training"]["batch_size"])
        mlflow.log_param("lr", config["training"]["lr"])
        mlflow.log_param("num_epochs", config["training"]["num_epochs"])
        mlflow.log_param("num_workers", config["training"]["num_workers"])

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
