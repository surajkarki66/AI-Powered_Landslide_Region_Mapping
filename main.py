import typer
import torch
import yaml

from src.pipeline.train import train as train_model
from src.pipeline.cross_validation import run_cross_validation as cross_validate_model
from src.pipeline.export import export as export_model

torch.cuda.empty_cache()

app = typer.Typer()

def load_config():
    with open("configs/config.yaml", "r") as f:
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
def cross_validation():
    """
    Cross validate the landslide mapping model.
    """
    config = load_config()
    cross_validate_model(config['cross_validation'])

@app.command()
def export():
    """
    Export the landslide mapping model to ONNX format.
    """
    config = load_config()
    export_model(config['export'])

if __name__ == "__main__":
    app()
