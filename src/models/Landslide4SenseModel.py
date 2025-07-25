import os
import torch
import pandas as pd
import lightning as L
import segmentation_models_pytorch as smp

from src.pipeline.loss import get_loss_fn


class Landslide4SenseMappingModel(L.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, learning_rate, loss_function_name, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.loss_fn = get_loss_fn(loss_function_name)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.learning_rate = learning_rate
        # Metric logging for CSV
        self.metric_log_path = "assets/training_metrics.csv"
        self.logged_metrics = {
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

    def forward(self, image):
        return self.model(image)

    def shared_step(self, batch, stage):
        image, mask = batch
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

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

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score": f1_score,
            f"{stage}_f2_score": f2_score,
            f"{stage}_accuracy": accuracy,
            f"{stage}_recall": recall,
            f"{stage}_precision": precision,
        }

        self.log_dict(metrics, prog_bar=True)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # Store current stage metrics
        self.logged_metrics["epoch"].append(self.current_epoch)
        self.logged_metrics["stage"].append(stage)
        self.logged_metrics["per_image_iou"].append(per_image_iou.item())
        self.logged_metrics["dataset_iou"].append(dataset_iou.item())
        self.logged_metrics["f1_score"].append(f1_score.item())
        self.logged_metrics["f2_score"].append(f2_score.item())
        self.logged_metrics["accuracy"].append(accuracy.item())
        self.logged_metrics["recall"].append(recall.item())
        self.logged_metrics["precision"].append(precision.item())
        self.logged_metrics["loss"].append(avg_loss.item())

        # Save just this stage's metrics to CSV
        self.save_epoch_metrics_to_csv()

    def save_epoch_metrics_to_csv(self):
        if not self.logged_metrics["epoch"]:
            return

        df = pd.DataFrame(self.logged_metrics)
        file_exists = os.path.exists(self.metric_log_path)

        if not file_exists:
            df.to_csv(self.metric_log_path, index=False)
        else:
            df.to_csv(self.metric_log_path, mode="a", header=False, index=False)

        # Clear after saving this stage
        for key in self.logged_metrics:
            self.logged_metrics[key].clear()

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
