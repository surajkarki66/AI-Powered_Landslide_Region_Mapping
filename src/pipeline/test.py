import os
import json
import torch
import lightning as L
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from utils.dataset import Landslide_Dataset
from model import LandslideMappingModel


def test(config):
    # data paths
    x_test_dir = os.path.join(config["data_path"], "test", "img")
    y_test_dir = os.path.join(config["data_path"], "test", "mask")

    # dataset
    test_dataset = Landslide_Dataset(x_test_dir, y_test_dir)

    # data loader
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # model
    model = LandslideMappingModel(
        config["arch"],
        config["encoder_name"],
        in_channels=config["in_channels"],
        out_classes=config["out_classes"],
    )
    model.load_state_dict(torch.load(config["model_output_path"]))

    # trainer
    trainer = L.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["default_root_dir"],
    )

    # test
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)

    # Save metrics to a JSON file
    metrics_output_path = config.get("metrics_output_path", "metrics.json")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    with open(metrics_output_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("Test metrics saved to", metrics_output_path)

    # visualize predictions
    visualize_predictions(model, test_loader)


def visualize_predictions(model, test_loader):
    os.makedirs("assets/plots", exist_ok=True)
    images, masks = next(iter(test_loader))
    with torch.inference_mode():
        model.eval()
        logits = model(images)
    pr_masks = logits.sigmoid()
    for idx, (image, gt_mask, pr_mask) in enumerate(zip(images, masks, pr_masks)):
        if idx <= 4:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.title("Image")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask.numpy().squeeze())
            plt.title("Ground truth")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(pr_mask.numpy().squeeze())
            plt.title("Prediction")
            plt.axis("off")
            plt.savefig(f"assets/plots/prediction_{idx}.png")
            plt.close()
        else:
            break