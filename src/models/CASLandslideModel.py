import os
import torch
import pandas as pd
import lightning as L
import segmentation_models_pytorch as smp

from src.pipeline.loss import get_loss_fn


class CASLandslideMappingModel(L.LightningModule):
    """
    PyTorch Lightning Module for Landslide Segmentation using SMP (segmentation_models_pytorch).
    Includes preprocessing normalization, loss computation, and evaluation metrics (IoU, F1, Recall, etc.).
    """

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str | None,
        in_channels: int,
        out_classes: int,
        learning_rate: float,
        loss_function_name: str,
        **kwargs
    ):
        """
        Initialize CAS Landslide Segmentation Model.

        Args:
            arch (str): Model architecture (e.g., "Unet", "FPN").
            encoder_name (str): Encoder backbone (e.g., "resnet34").
            encoder_weights (str | None): Pretrained weights (e.g., "imagenet").
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            out_classes (int): Number of output classes.
            learning_rate (float): Optimizer learning rate.
            loss_function_name (str): Loss function identifier.
            **kwargs: Extra arguments for SMP model creation.
        """
        super().__init__()
        # Build SMP model
        self.model: torch.nn.Module = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.learning_rate: float = learning_rate

        # Get preprocessing parameters for input normalization
        params: dict = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function
        self.loss_fn = get_loss_fn(loss_function_name)

        # Temporary storage for outputs
        self.training_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

        # Metrics log configuration
        self.metric_log_path: str = "assets/training_metrics.csv"
        self.logged_metrics: dict[str, list] = {
            "epoch": [],
            "stage": [],
            "loss": [],
            "per_image_iou": [],
            "dataset_iou": [],
            "f1_score": [],
            "f2_score": [],
            "accuracy": [],
            "recall": [],
            "precision": [],
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model with normalization applied.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Output segmentation logits.
        """
        # Apply normalization before inference
        image = (image - self.mean) / self.std
        mask: torch.Tensor = self.model(image)
        return mask

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
        """
        Shared logic for train/val/test steps:
        - forward pass
        - loss computation
        - stats collection for metrics

        Args:
            batch (tuple): (images, masks)
            stage (str): "train" | "valid" | "test"
        """
        image, mask = batch

        # Input shape validation
        assert image.ndim == 4, "Image tensor must be 4D (B, C, H, W)."
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0, "Image dimensions must be divisible by 32."
        assert mask.ndim == 4, "Mask tensor must be 4D (B, 1, H, W)."
        assert mask.max() <= 1.0 and mask.min() >= 0, "Mask values must be in [0, 1]."

        # Forward pass
        logits_mask: torch.Tensor = self.forward(image)
        loss: torch.Tensor = self.loss_fn(logits_mask, mask)

        # Threshold predictions
        prob_mask: torch.Tensor = logits_mask.sigmoid()
        pred_mask: torch.Tensor = (prob_mask > 0.5).float()

        # Compute statistics for metrics
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs: list[dict[str, torch.Tensor]], stage: str) -> None:
        """
        Compute aggregated metrics after each epoch.

        Args:
            outputs (list): Collected step outputs (loss + stats).
            stage (str): "train" | "valid" | "test"
        """
        # Concatenate stats from batches
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Compute metrics
        per_image_iou: torch.Tensor = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou: torch.Tensor = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score: torch.Tensor = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score: torch.Tensor = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy: torch.Tensor = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall: torch.Tensor = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision: torch.Tensor = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")

        metrics: dict[str, torch.Tensor] = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score": f1_score,
            f"{stage}_f2_score": f2_score,
            f"{stage}_accuracy": accuracy,
            f"{stage}_recall": recall,
            f"{stage}_precision": precision,
        }

        # Log to Lightning
        self.log_dict(metrics, prog_bar=True)

        # Average loss
        avg_loss: torch.Tensor = torch.stack([x["loss"] for x in outputs]).mean()

        # Save metrics in dictionary
        self.logged_metrics["epoch"].append(int(self.current_epoch))
        self.logged_metrics["stage"].append(stage)
        self.logged_metrics["per_image_iou"].append(float(per_image_iou.item()))
        self.logged_metrics["dataset_iou"].append(float(dataset_iou.item()))
        self.logged_metrics["f1_score"].append(float(f1_score.item()))
        self.logged_metrics["f2_score"].append(float(f2_score.item()))
        self.logged_metrics["accuracy"].append(float(accuracy.item()))
        self.logged_metrics["recall"].append(float(recall.item()))
        self.logged_metrics["precision"].append(float(precision.item()))
        self.logged_metrics["loss"].append(float(avg_loss.item()))

        # Save metrics to CSV
        self.save_epoch_metrics_to_csv()

    def save_epoch_metrics_to_csv(self) -> None:
        """Append current epoch metrics to CSV file and reset buffers."""
        if not self.logged_metrics["epoch"]:
            return

        df: pd.DataFrame = pd.DataFrame(self.logged_metrics)
        file_exists: bool = os.path.exists(self.metric_log_path)

        if not file_exists:
            df.to_csv(self.metric_log_path, index=False)
        else:
            df.to_csv(self.metric_log_path, mode="a", header=False, index=False)

        # Clear buffer after saving
        for key in self.logged_metrics:
            self.logged_metrics[key].clear()

    # ------------------- Training / Validation / Test hooks -------------------

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single training step."""
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single validation step."""
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single test step."""
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> dict:
        """
        Configure optimizer (Adam) and scheduler (CosineAnnealingLR).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10 * 100, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update every step
                "frequency": 1,
            },
        }
