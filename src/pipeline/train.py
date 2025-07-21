import os
import json
import torch
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.utils.dataset import CAS_Landslide_Dataset
from src.model import CASLandslideMappingModel

def train(config):
    # data paths
    x_train_dir = os.path.join(config["data_path"], "train", "img")
    y_train_dir = os.path.join(config["data_path"], "train", "mask")
    x_valid_dir = os.path.join(config["data_path"], "val", "img")
    y_valid_dir = os.path.join(config["data_path"], "val", "mask")

    # datasets
    train_dataset = CAS_Landslide_Dataset(x_train_dir, y_train_dir)
    valid_dataset = CAS_Landslide_Dataset(x_valid_dir, y_valid_dir)

    # data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # model
    model = CASLandslideMappingModel(
        config["arch"],
        config["encoder_name"],
        config["encoder_weights"],
        in_channels=config["in_channels"],
        out_classes=config["out_classes"],
        learning_rate=config["learning_rate"],
        loss_function_name=config["loss_function"],
    )

    # trainer
    early_stopping = EarlyStopping(
        monitor=config["early_stopping"]["monitor"],
        mode=config["early_stopping"]["mode"],
        patience=config["early_stopping"]["patience"],
    )

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        log_every_n_steps=1,
        callbacks=[early_stopping],
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["default_root_dir"],
    )

    # fit the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # plot metrics
    plot_metrics()

    # validate
    validate(trainer, model, valid_loader, config)

    # save model
    torch.save(model.state_dict(), config["model_output_path"])


def plot_metrics():
    os.makedirs("assets/plots", exist_ok=True)
    df = pd.read_csv("assets/training_metrics.csv")
    df = df.iloc[1:].reset_index(drop=True)
    df["epoch"] = df["epoch"].astype(int) + 1
    sns.set(style="whitegrid")
    metrics = [
        "loss",
        "per_image_iou",
        "dataset_iou",
        "f1_score",
        "f2_score",
        "accuracy",
        "recall",
        "precision",
    ]
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        for stage in ["train", "valid"]:
            stage_df = df[df["stage"] == stage]
            plt.plot(stage_df["epoch"], stage_df[metric], label=stage, linewidth=2)
        plt.title(f"{metric.replace('_', ' ').title()} vs Epoch", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.xticks(sorted(df["epoch"].unique()))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"assets/plots/metrics_{metric}.png")
        plt.close()


def validate(trainer, model, valid_loader, config):
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)

    # Save metrics to a JSON file
    metrics_output_path = config.get("metrics_output_path", "metrics.json")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    with open(metrics_output_path, "w") as f:
        json.dump(valid_metrics, f, indent=4)

    print("Validation metrics saved to", metrics_output_path)

