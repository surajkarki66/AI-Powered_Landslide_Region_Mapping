import segmentation_models_pytorch as smp
from torch.nn.modules.loss import _Loss


def get_loss_fn(loss_name: str) -> _Loss:
    """
    Return a loss function from segmentation_models_pytorch based on the given name.

    Parameters
    ----------
    loss_name : str
        The name of the loss function. Supported values are:
        "DiceLoss", "JaccardLoss", "TverskyLoss", "FocalLoss", "SoftBCE", "MCCLoss".

    Returns
    -------
    _Loss
        A PyTorch loss function object.
    """

    if loss_name == "DiceLoss":
        # Dice Loss for binary segmentation
        return smp.losses.DiceLoss(smp.losses.BINARY_MODE, log_loss=False, from_logits=True)

    elif loss_name == "JaccardLoss":
        # Jaccard (IoU) Loss for binary segmentation
        return smp.losses.JaccardLoss(smp.losses.BINARY_MODE, log_loss=False, from_logits=True)

    elif loss_name == "TverskyLoss":
        # Tversky Loss (generalized Dice) with alpha=0.5, beta=0.5
        return smp.losses.TverskyLoss(
            smp.losses.BINARY_MODE,
            log_loss=False,
            from_logits=True,
            alpha=0.5,
            beta=0.5,
            gamma=1.0
        )

    elif loss_name == "FocalLoss":
        # Focal Loss with gamma=2.0 for binary segmentation
        return smp.losses.FocalLoss(smp.losses.BINARY_MODE, gamma=2.0)

    elif loss_name == "SoftBCE":
        # Soft Binary Cross Entropy with logits
        return smp.losses.SoftBCEWithLogitsLoss()

    elif loss_name == "MCCLoss":
        # Matthews Correlation Coefficient Loss
        return smp.losses.MCCLoss()

    else:
        # Raise error if unsupported loss is requested
        raise ValueError(f"Unknown loss function: {loss_name}")
