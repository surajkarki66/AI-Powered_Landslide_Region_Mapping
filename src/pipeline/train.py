import os
import json
import segmentation_models_pytorch as smp
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Dict, Any, Optional, List
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.utils.dataset import CAS_Landslide_Dataset, Landslide4SenseDataset
from src.models.CASLandslideModel import CASLandslideMappingModel
from src.models.Landslide4SenseModel import Landslide4SenseMappingModel


def train(config: Dict[str, Any]) -> None:
    """
    Train, validate, and test a landslide mapping model.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing all
            parameters like paths, hyperparameters, and model settings.
    """

    # Define dataset paths
    x_train_dir: str = os.path.join(config["data_path"], "train", "img")
    y_train_dir: str = os.path.join(config["data_path"], "train", "mask")
    x_valid_dir: str = os.path.join(config["data_path"], "val", "img")
    y_valid_dir: str = os.path.join(config["data_path"], "val", "mask")
    x_test_dir: str = os.path.join(config["data_path"], "test", "img")
    y_test_dir: str = os.path.join(config["data_path"], "test", "mask")

    # Dataset placeholders (type hinting for clarity)
    train_dataset: Optional[object] = None
    valid_dataset: Optional[object] = None
    test_dataset: Optional[object] = None

    model_type: str = config.get("model_type")
    if model_type == "CASLandslide":
        # Model initialization
        model: CASLandslideMappingModel = CASLandslideMappingModel(
            config["arch"],
            config["encoder_name"],
            config["encoder_weights"],
            in_channels=config["in_channels"],
            out_classes=config["out_classes"],
            learning_rate=config["learning_rate"],
            loss_function_name=config["loss_function"],
        )
        # Datasets
        train_dataset = CAS_Landslide_Dataset(x_train_dir, y_train_dir)
        valid_dataset = CAS_Landslide_Dataset(x_valid_dir, y_valid_dir)
        test_dataset = CAS_Landslide_Dataset(x_test_dir, y_test_dir)

    elif model_type == "Landslide4Sense":
        # Model initialization
        model: Landslide4SenseMappingModel = Landslide4SenseMappingModel(
            config["arch"],
            config["encoder_name"],
            config["encoder_weights"],
            in_channels=config["in_channels"],
            out_classes=config["out_classes"],
            learning_rate=config["learning_rate"],
            loss_function_name=config["loss_function"],
        )
        # Datasets
        train_dataset = Landslide4SenseDataset(x_train_dir, y_train_dir)
        valid_dataset = Landslide4SenseDataset(x_valid_dir, y_valid_dir)
        test_dataset = Landslide4SenseDataset(x_test_dir, y_test_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    valid_loader: DataLoader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    test_loader: DataLoader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    early_stopping: EarlyStopping = EarlyStopping(
        monitor=config["early_stopping"]["monitor"],
        mode=config["early_stopping"]["mode"],
        patience=config["early_stopping"]["patience"],
    )

    trainer: L.Trainer = L.Trainer(
        max_epochs=config["epochs"],
        log_every_n_steps=1,
        callbacks=[early_stopping],
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["default_root_dir"],
        strategy="ddp_find_unused_parameters_true",
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    # Plot training/validation metrics
    plot_metrics()

    # Run validation
    validate(trainer, model, valid_loader, config)
    
    # Run testing
    test(trainer, model, test_loader, config)

    model.model.save_pretrained(config["model_output_path"], dataset=config["model_type"])
    restored_model = smp.from_pretrained(config["model_output_path"])
    
    if restored_model:
        print(f"Model is saved at {config['model_output_path']}")


def plot_metrics() -> None:
    """
    Plot training and validation metrics saved in CSV.
    Saves figures into assets/plots directory.
    """
    os.makedirs("assets/plots", exist_ok=True)

    # Load metrics CSV
    df: pd.DataFrame = pd.read_csv("assets/training_metrics.csv")
    df = df.iloc[1:].reset_index(drop=True)
    df["epoch"] = df["epoch"].astype(int) + 1

    sns.set(style="whitegrid")

    # Metrics to visualize
    metrics: List[str] = [
        "loss",
        "per_image_iou",
        "dataset_iou",
        "f1_score",
        "f2_score",
        "accuracy",
        "recall",
        "precision",
    ]

    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        for stage in ["train", "valid"]:
            stage_df: pd.DataFrame = df[df["stage"] == stage]
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


def validate(trainer: L.Trainer, model: Any, valid_loader: DataLoader, config: Dict[str, Any]) -> None:
    """
    Run validation on the model and save metrics to JSON.
    """
    valid_metrics: List[Dict[str, Any]] = trainer.validate(model, dataloaders=valid_loader, verbose=True)

    metrics_output_path: str = "assets/validation_metrics.json"
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    with open(metrics_output_path, "w") as f:
        json.dump(valid_metrics, f, indent=4)

    print("Validation metrics saved to", metrics_output_path)
    
    
def test(trainer: L.Trainer, model: Any, test_loader: DataLoader, config: Dict[str, Any]) -> None:
    """
    Run testing on the model and save metrics to JSON.
    """
    test_metrics: List[Dict[str, Any]] = trainer.test(model, dataloaders=test_loader, verbose=True)

    metrics_output_path: str = "assets/test_metrics.json"
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    with open(metrics_output_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("Test metrics saved to", metrics_output_path)
