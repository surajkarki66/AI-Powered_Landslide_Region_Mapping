import torch
import onnx
import onnxruntime
import numpy as np
import segmentation_models_pytorch as smp
from typing import Dict, Tuple, Any


def export(config: Dict[str, Any]) -> None:
    """
    Full pipeline for exporting a PyTorch model to ONNX format, 
    validating the export, running inference with ONNX Runtime, 
    and verifying results against the original PyTorch model.
    
    Args:
        config (dict): Configuration dictionary containing:
            - model_path (str): Path to pretrained model.
            - input_shape (list[int] or tuple[int]): Example input shape (e.g., [1, 3, 256, 256]).
            - onnx_model_output_path (str): Path to save exported ONNX model.
    """
    # Load pretrained model (segmentation_models_pytorch)
    model = smp.from_pretrained(config["model_path"])
    model.eval()

    # Step 1: Export to ONNX
    export_to_onnx(model, config)

    # Step 2: Validate exported ONNX model
    check_with_onnx(config)

    # Step 3: Run inference using ONNX Runtime
    run_with_onnxruntime(config)

    # Step 4: Verify ONNX output matches PyTorch output
    verify_with_pytorch_model(model, config)


def export_to_onnx(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
        config (dict): Configuration dictionary.
    """
    input_shape: Tuple[int, ...] = tuple(config["input_shape"])  # Ensure tuple
    dynamic_axes = {0: "batch_size", 2: "height", 3: "width"}   # Allow dynamic input size

    # Export model
    torch.onnx.export(
        model=model,
        args=torch.randn(input_shape),  # Dummy input for tracing
        f=config["onnx_model_output_path"],  # Output ONNX file
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": dynamic_axes, "output": dynamic_axes},
    )
    print(f"✅ Model exported to {config['onnx_model_output_path']}")


def check_with_onnx(config: Dict[str, Any]) -> None:
    """
    Validate exported ONNX model.
    
    Args:
        config (dict): Configuration dictionary.
    """
    onnx_model = onnx.load(config["onnx_model_output_path"])
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model check passed!")


def run_with_onnxruntime(config: Dict[str, Any]) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Run inference using ONNX Runtime.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        Tuple[np.ndarray, torch.Tensor]: (ONNX outputs, input tensor used)
    """
    input_shape: Tuple[int, ...] = tuple(config["input_shape"])
    sample: torch.Tensor = torch.randn(input_shape, dtype=torch.float32)  # Ensure float32

    ort_session = onnxruntime.InferenceSession(
        config["onnx_model_output_path"], providers=["CPUExecutionProvider"]
    )

    ort_inputs = {"input": sample.numpy()}
    ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)

    print("✅ ONNX Runtime inference successful!")
    return ort_outputs, sample


def verify_with_pytorch_model(model: torch.nn.Module, config: Dict[str, Any]) -> None:
    """
    Compare ONNX Runtime inference results with PyTorch model outputs.
    
    Args:
        model (torch.nn.Module): The original PyTorch model.
        config (dict): Configuration dictionary.
    """
    ort_outputs, sample = run_with_onnxruntime(config)

    with torch.inference_mode():
        torch_out: torch.Tensor = model(sample)

    # Convert PyTorch tensor to NumPy for comparison
    np.testing.assert_allclose(
        torch_out.detach().cpu().numpy(),
        ort_outputs[0],
        rtol=1e-03,
        atol=1e-05,
    )
    print("✅ ONNX and PyTorch model outputs match within tolerance!")


