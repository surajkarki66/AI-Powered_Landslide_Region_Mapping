import typer
import yaml

from src.pipeline.train import train as train_model
from src.pipeline.test import test as test_model
from src.pipeline.export import export as export_model

import torch
torch.cuda.empty_cache()

app = typer.Typer()

def load_config():
    with open("configs/config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config

@app.command()
def train():
    """
    Train the landslide mapping model.
    """
    config = load_config()
    train_model(config['train'])

@app.command()
def test():
    """
    Test the landslide mapping model.
    """
    config = load_config()
    test_model(config['test'])

@app.command()
def export():
    """
    Export the landslide mapping model to ONNX format.
    """
    config = load_config()
    export_model(config['export'])

if __name__ == "__main__":
    app()
