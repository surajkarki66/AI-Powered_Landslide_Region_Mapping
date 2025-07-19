import segmentation_models_pytorch as smp



def get_loss_fn(loss_name):
    if loss_name == "DiceLoss":
        return smp.losses.DiceLoss(smp.losses.BINARY_MODE, log_loss=False, from_logits=True)
    elif loss_name == "JaccardLoss":
        return smp.losses.JaccardLoss(smp.losses.BINARY_MODE, log_loss=False, from_logits=True)
    elif loss_name == "TverskyLoss":
        return smp.losses.TverskyLoss(smp.losses.BINARY_MODE, log_loss=False, from_logits=True, alpha=0.5, beta=0.5, gamma=1.0)
    elif loss_name == "FocalLoss":
        return smp.losses.FocalLoss(smp.losses.BINARY_MODE, gamma=2.0)
    elif loss_name == "SoftBCE":
        return smp.losses.SoftBCEWithLogitsLoss()
    elif loss_name == "MCCLoss":
        return smp.losses.MCCLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

