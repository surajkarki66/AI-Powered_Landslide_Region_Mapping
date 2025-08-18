import os
import pandas as pd
import torch
import lightning as L

from typing import Dict, Any, List
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.Landslide4SenseModel import Landslide4SenseMappingModel
from src.models.CASLandslideModel import CASLandslideMappingModel
from src.utils.dataset import (
    CAS_Landslide_Dataset_Cross_Validation,
    Landslide4SenseDataset_CrossValidation,
)


def run_cross_validation(config: Dict[str, Any]) -> None:
    """
    Run K-Fold cross-validation for CAS Landslide or Landslide4Sense datasets.

    Parameters
    ----------
    config : dict
        Dictionary containing training configuration:
        {
            "images_dir": str,
            "masks_dir": str,
            "num_folds": int,
            "model_type": str,               # "CASLandslide" or "Landslide4Sense"
            "batch_size": int,
            "arch": str,                     # segmentation model architecture
            "encoder_name": str,
            "encoder_weights": str | None,
            "in_channels": int,
            "out_classes": int,
            "learning_rate": float,
            "loss_function": str,
            "epochs": int,
            "save_dir": str,
            "early_stopping": {
                "monitor": str,
                "mode": str,
                "patience": int
            }
        }
    """

    # ✅ Ensure reproducibility
    L.seed_everything(42, workers=True)

    images_dir: str = config.get("images_dir")
    masks_dir: str = config.get("masks_dir")
    num_folds: int = int(config.get("num_folds", 5))
    model_type: str = config.get("model_type", "CASLandslide")
    batch_size: int = int(config.get("batch_size", 8))
    arch: str = config.get("arch", "FPN")
    encoder_name: str = config.get("encoder_name", "mobilenet_v2")
    encoder_weights: str | None = config.get("encoder_weights", "imagenet")
    in_channels: int = int(config.get("in_channels", 3))
    out_classes: int = int(config.get("out_classes", 2))
    learning_rate: float = float(config.get("learning_rate", 1e-3))
    loss_function_name: str = config.get("loss_function", "DiceLoss")
    epochs: int = int(config.get("epochs", 50))
    save_dir: str = config.get("save_dir", "./results")

    # ✅ Gather all image IDs
    all_ids: List[str] = sorted(os.listdir(images_dir))

    # ✅ Prepare KFold splitter
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    summary_rows: List[Dict[str, Any]] = []

    # 🚀 Start cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_ids)):
        print(f"\n🔁 Fold {fold + 1}/{num_folds}")

        train_dataset: Dataset
        val_dataset: Dataset
        model: L.LightningModule

        if model_type == "CASLandslide":
            train_dataset = CAS_Landslide_Dataset_Cross_Validation(images_dir, masks_dir, indices=train_idx)
            val_dataset = CAS_Landslide_Dataset_Cross_Validation(images_dir, masks_dir, indices=val_idx)
            model = CASLandslideMappingModel(
                arch, encoder_name, encoder_weights, in_channels, out_classes, learning_rate, loss_function_name
            )
        else:  # Landslide4Sense
            train_dataset = Landslide4SenseDataset_CrossValidation(images_dir, masks_dir, indices=train_idx)
            val_dataset = Landslide4SenseDataset_CrossValidation(images_dir, masks_dir, indices=val_idx)
            model = Landslide4SenseMappingModel(
                arch, encoder_name, encoder_weights, in_channels, out_classes, learning_rate, loss_function_name
            )


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        early_stopping = EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            mode=config["early_stopping"]["mode"],
            patience=int(config["early_stopping"]["patience"]),
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join("checkpoints", f"fold_{fold + 1}"),
            filename="{epoch}-{val_loss:.4f}",
            save_top_k=1,
            monitor=config["early_stopping"]["monitor"],
            mode=config["early_stopping"]["mode"],
        )

        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            max_epochs=epochs,
            callbacks=[early_stopping, checkpoint_callback],
            log_every_n_steps=10,
        )

        trainer.fit(model, train_loader, val_loader)

        test_metrics: Dict[str, Any] = trainer.test(model, dataloaders=val_loader, verbose=True)[0]
        print(f"✅ Fold {fold + 1} Completed.")

        # Save results for this fold
        row: Dict[str, Any] = {
            "fold": fold + 1,
            **test_metrics
        }
        summary_rows.append(row)

    if summary_rows:
        df: pd.DataFrame = pd.DataFrame(summary_rows)
        results_path: str = os.path.join(save_dir, f"crossval_results_{model_type}.csv")
        df.to_csv(results_path, index=False)
        print(f"\n📊 Fold summary saved to: {results_path}")

        mean_metrics = df.mean(numeric_only=True)
        std_metrics = df.std(numeric_only=True)

        print("\n📈 Cross-validation summary (mean ± std):")
        for col in mean_metrics.index:
            print(f"{col}: {mean_metrics[col]:.4f} ± {std_metrics[col]:.4f}")
