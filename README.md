# AI-Powered Landslide Region Mapping using Semantic Segmentation

## Overview

This project leverages deep learning and semantic segmentation to automatically map landslide regions from satellite and UAV imagery. It aims to support disaster management and geo-spatial analysis by providing accurate landslide detection using state-of-the-art models.

## Features

- Semantic segmentation of landslide regions in satellite/UAV images
- Modular pipeline for data preprocessing, model training, and evaluation
- Visualization of results and metrics
- Support for multiple benchmark datasets

## Directory Structure

```
├── app.py                  # Streamlit application entry point
├── main.py                 # Main script for running experiments/training
├── src/                    # Source code
│   ├── models/             # Model definitions
│   ├── pipeline/           # Data and training pipeline
│   ├── utils/              # Utility functions
│   └── views/              # Visualization and reporting
├── assets/                 # Generated plots, metrics, and trained models
├── configs/                # Configuration files
├── data/                   # Datasets and test data
├── experiments/            # Notebooks for data analysis and model training
├── docs/                   # Documentation and presentations
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── LICENSE                 # License information
└── README.md               # Project overview
```

## Data Sources

This project uses publicly available datasets:

- **CAS Landslide Dataset**
  Xu, Y., Ouyang, C., Xu, Q., Wang, D., Zhao, B., & Luo, Y. (2023). *CAS Landslide Dataset: A Large-Scale and Multisensor Dataset for Deep Learning-Based Landslide Detection* [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.10294997](https://doi.org/10.5281/zenodo.10294997)
- **Landslide4Sense Dataset**
  Ghorbanzadeh, O., Xu, Y., Ghamisi, P., Kopp, M., & Kreil, D. (2022). *Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection* [Data set]. *IEEE Transactions on Geoscience and Remote Sensing, 60*, 1–17. Zenodo. [https://doi.org/10.5281/zenodo.10463239](https://doi.org/10.5281/zenodo.10463239)

Place your training, validation, and testing data in the `data/` directory.

## Getting Started

**Install dependencies:**

```sh
pip install -r requirements.txt
```

---

## Instructions

### Training the Model

To train the semantic segmentation model:

1. **Prepare data:**

   - Download datasets and place them in `data/`.
2. **Run**

```sh
python main.py train
```

- Adjust the configuration files in `configs/` as needed.
- Training outputs (metrics, models) will be saved in the `assets/` directory.

### Cross validating the Model

This project supports **K-Fold Cross-Validation** to evaluate model robustness across multiple dataset splits.

```sh
python main.py cross-validation
```

### Exporting the Model

After training or cross-validation, you can export the trained PyTorch models to ONNX for deployment.

```sh
python main.py export
```

### Running the Application

First, you need to download all the [pre-trained models](https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping_using_Semantic_Segmentation/releases/tag/LandslideSegmentationModels_v1.0) and placed them inside the assets directory.

To launch the interactive Streamlit app for landslide region mapping:

```sh
python -m streamlit run app.py
```

- The app allows you to upload images, run inference, and visualize segmentation results.
- Make sure your trained model is available in the expected location (see config).

---

## Results & Assets

- Training, validation, and testing metrics are stored in `assets/`.
- Plots and visualizations are saved in `assets/plots/`.

## Pretrained Models

You can download the pretrained models used in this project: [Download from here](https://github.com/surajkarki66/AI-Powered_Landslide_Region_Mapping_using_Semantic_Segmentation/releases/tag/LandslideSegmentationModels_v1.0)
