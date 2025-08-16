import os
import pandas as pd
import torch
import lightning as L
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from src.models.Landslide4SenseModel import Landslide4SenseMappingModel
from src.models.CASLandslideModel import CASLandslideMappingModel
from src.utils.dataset import CAS_Landslide_Dataset_Cross_Validation, Landslide4SenseDataset_CrossValidation

def run_cross_validation(config):
    images_dir = config.get("images_dir")
    masks_dir = config.get("masks_dir")
    num_folds = config.get("num_folds")
    model_type = config.get("model_type")
    batch_size = config.get("batch_size")
    arch = config.get("arch")
    encoder_name = config.get("encoder_name")
    encoder_weights = config.get("encoder_weights")
    in_channels = config.get("in_channels")
    out_classes = config.get("out_classes")
    learning_rate = config.get("learning_rate")
    loss_function_name = config.get("loss_function")
    epochs = config.get("epochs")
    save_dir = config.get("save_dir")
    
    
    all_ids = sorted(os.listdir(images_dir))
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    summary_rows = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_ids)):
        print(f"\n🔁 Fold {fold + 1}/{num_folds}")

        train_dataset = None
        test_dataset = None
        model = None
        
        if model_type == "CASLandslide":
            train_dataset = CAS_Landslide_Dataset_Cross_Validation(images_dir, masks_dir, indices=train_idx)
            test_dataset = CAS_Landslide_Dataset_Cross_Validation(images_dir, masks_dir, indices=val_idx)
            model = CASLandslideMappingModel(arch, encoder_name, encoder_weights, in_channels, out_classes, learning_rate, loss_function_name)

        else:
            train_dataset = Landslide4SenseDataset_CrossValidation(images_dir, masks_dir, indices=train_idx)
            test_dataset = Landslide4SenseDataset_CrossValidation(images_dir, masks_dir, indices=val_idx)
            model = Landslide4SenseMappingModel(arch, encoder_name, encoder_weights, in_channels, out_classes, learning_rate, loss_function_name)

        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        early_stopping = EarlyStopping(
            monitor=config["early_stopping"]["monitor"],
            mode=config["early_stopping"]["mode"],
            patience=config["early_stopping"]["patience"],
        )
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            max_epochs=epochs,
            callbacks=[early_stopping],
            log_every_n_steps=10,
        )

        trainer.fit(model, train_loader, test_loader)
        test_metrics = trainer.test(model, dataloaders=test_loader, verbose=True)[0]
        print(f"✅ Fold {fold + 1} Completed.")
        row = {
            "fold": fold + 1,
            **test_metrics
        }
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(save_dir, index=False)
        print(f"\n📊 Fold summary saved to: {save_dir}")
