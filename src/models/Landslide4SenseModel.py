import os
import torch
import pandas as pd
import lightning as L
import segmentation_models_pytorch as smp

from src.pipeline.loss import get_loss_fn


class Landslide4SenseMappingModel(L.LightningModule):
    """
    PyTorch Lightning Module for Landslide Segmentation using segmentation_models_pytorch (SMP).
    Tracks and logs metrics (IoU, F1, Recall, etc.) during training, validation, and testing.
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
        Initialize the Landslide4Sense segmentation model.

        Args:
            arch (str): Model architecture (e.g., "FPN", "Unet").
            encoder_name (str): Encoder backbone name.
            encoder_weights (str | None): Pretrained weights (e.g., "imagenet") or None.
            in_channels (int): Number of input channels.
            out_classes (int): Number of output classes (segmentation classes).
            learning_rate (float): Learning rate for optimizer.
            loss_function_name (str): Name of the loss function to use.
            **kwargs: Additional arguments for SMP model creation.
        """
        super().__init__()
        self.model: torch.nn.Module = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.loss_fn = get_loss_fn(loss_function_name)

        # Storage for outputs during steps
        self.training_step_outputs: list[dict[str, torch.Tensor]] = []
        self.validation_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_step_outputs: list[dict[str, torch.Tensor]] = []

        self.learning_rate: float = learning_rate

        # CSV logging setup
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
        """Forward pass through the segmentation model."""
        return self.model(image)

    def shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> dict[str, torch.Tensor]:
        """
        Shared training/validation/testing step.
        Computes predictions, loss, and confusion matrix stats.
        """
        image, mask = batch
        assert image.ndim == 4  # Expect shape (B, C, H, W)
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0  # Ensure dimensions divisible by 32 (SMP requirement)
        assert mask.ndim == 4  # Expect shape (B, 1, H, W)
        assert mask.max() <= 1.0 and mask.min() >= 0  # Ensure binary masks in [0, 1]

        logits_mask: torch.Tensor = self.forward(image)
        loss: torch.Tensor = self.loss_fn(logits_mask, mask)

        # Convert logits to probabilities → binary predictions
        prob_mask: torch.Tensor = logits_mask.sigmoid()
        pred_mask: torch.Tensor = (prob_mask > 0.5).float()

        # Compute confusion matrix stats
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs: list[dict[str, torch.Tensor]], stage: str) -> None:
        """
        Aggregate metrics at the end of an epoch (for train/val/test).
        Logs IoU, F1, accuracy, recall, precision, etc.
        """
        # Concatenate statistics from all batches
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Compute evaluation metrics
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

        # Log metrics to Lightning progress bar
        self.log_dict(metrics, prog_bar=True)

        # Average loss across epoch
        avg_loss: torch.Tensor = torch.stack([x["loss"] for x in outputs]).mean()

        # Store metrics for CSV logging
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
        """Save epoch metrics to CSV file."""
        if not self.logged_metrics["epoch"]:
            return  # Skip if no metrics recorded

        df: pd.DataFrame = pd.DataFrame(self.logged_metrics)
        file_exists: bool = os.path.exists(self.metric_log_path)

        # Write or append to CSV
        if not file_exists:
            df.to_csv(self.metric_log_path, index=False)
        else:
            df.to_csv(self.metric_log_path, mode="a", header=False, index=False)

        # Clear after saving this stage
        for key in self.logged_metrics:
            self.logged_metrics[key].clear()

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single training step."""
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics at epoch end."""
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single validation step."""
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at epoch end."""
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        """Perform a single test step."""
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at epoch end."""
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> dict:
        """Configure optimizer (Adam) and scheduler (CosineAnnealingLR)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
            },
        }
