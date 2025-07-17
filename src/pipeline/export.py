import torch
import onnx
import onnxruntime
import numpy as np
from model import LandslideMappingModel

def export(config):
    # model to export
    model = LandslideMappingModel(
        config["arch"],
        config["encoder_name"],
        in_channels=config["in_channels"],
        out_classes=config["out_classes"],
    )

    # load trained model
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()

    # export to onnx
    export_to_onnx(model, config)

    # check with onnx
    check_with_onnx(config)

    # run with onnxruntime
    run_with_onnxruntime(config)

    # verify it's the same as for pytorch model
    verify_with_pytorch_model(model, config)


def export_to_onnx(model, config):
    input_shape = tuple(config["input_shape"])
    dynamic_axes = {0: "batch_size", 2: "height", 3: "width"}
    torch.onnx.export(
        model,  # model being run
        torch.randn(input_shape),  # model input
        config["onnx_model_output_path"],  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=17,  # the ONNX version to export
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={  # variable length axes
            "input": dynamic_axes,
            "output": dynamic_axes,
        },
    )


def check_with_onnx(config):
    onnx_model = onnx.load(config["onnx_model_output_path"])
    onnx.checker.check_model(onnx_model)


def run_with_onnxruntime(config):
    input_shape = tuple(config["input_shape"])
    sample = torch.randn(input_shape)
    ort_session = onnxruntime.InferenceSession(
        config["onnx_model_output_path"], providers=["CPUExecutionProvider"]
    )
    ort_inputs = {"input": sample.numpy()}
    ort_outputs = ort_session.run(output_names=None, input_feed=ort_inputs)
    return ort_outputs, sample


def verify_with_pytorch_model(model, config):
    ort_outputs, sample = run_with_onnxruntime(config)
    with torch.inference_mode():
        torch_out = model(sample)
    np.testing.assert_allclose(torch_out.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
