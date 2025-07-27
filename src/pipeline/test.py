import os
import json
import torch
import lightning as L
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader

from src.utils.dataset import Landslide4SenseDataset, CAS_Landslide_Dataset_TIFF
from src.models.Landslide4SenseModel import Landslide4SenseMappingModel
from src.models.CASLandslideModel import CASLandslideMappingModel


def test(config):
    # data paths
    x_test_dir = os.path.join(config["data_path"], "test", "img")
    y_test_dir = os.path.join(config["data_path"], "test", "mask")

    # model selection
    model_type = config.get("model_type")
    if model_type == "CASLandslide":
        # datasets
        test_dataset = CAS_Landslide_Dataset_TIFF(x_test_dir, y_test_dir)

    elif model_type == "Landslide4Sense":
        # datasets
        test_dataset = Landslide4SenseDataset(x_test_dir, y_test_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # data loader
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    # load the model
    restored_model = smp.from_pretrained(config["model_output_path"])

    # trainer
    trainer = L.Trainer(
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=config["default_root_dir"],
    )

    # test
    test_metrics = trainer.test(restored_model, dataloaders=test_loader, verbose=False)

    # Save metrics to a JSON file
    metrics_output_path = config.get("metrics_output_path", "metrics.json")
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    with open(metrics_output_path, "w") as f:
        json.dump(test_metrics, f, indent=4)

    print("Test metrics saved to", metrics_output_path)

    # visualize predictions
    if model_type == "CASLandslide":
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
