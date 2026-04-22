# TL-TSC: Transfer Learning for Tree Species Composition Estimation across Boreal Forests

This repository provides the code, models, and workflows for the paper *"Assessing the Transferability of Multi-Source Deep Learning Models for Tree Species Composition Prediction across Boreal Forests"*.

It includes tools for fusing Single-Photon LiDAR (SPL) point clouds and Sentinel-2 satellite imagery, evaluating direct cross-site model transferability, analyzing ecological and data modality shifts (JSD, MMD², species dominance), and performing fine-tuning-based transfer learning (FTL) under limited inventory data budgets.

## Repository Structure

```text
data/
├── source_domain/                   # Training site datasets (e.g., RMF)
├── target_domain/                   # New site datasets (e.g., WRF)
dataset/                             # Dataloaders and augmentation scripts
models/                              # U-Net, PointNeXt, and fusion head architectures
utils/                               # Evaluation metrics and loss functions (EWMSE)
README.md
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yuwei-cao-git/TL-TSC.git  
    cd TL-TSC 
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt  
    ```

## Dependencies

  - Python 3.8+
  - PyTorch
  - PyTorch Lightning
  - RayTune
  - Open3D
  - Geopandas
  - Rasterio
  - Scikit-learn
  - SciPy (for distributional shift metrics)
  - GDAL

## Example Use Case

1.  Run 'data_processing/superpixel_gen.sh' to generate dataset
2.  Run 'superpixel_resplit_5.py' to generate varying proportions of dataset (1% to 100%)
3.  Run `train_fuse.py` to train the multi-modal fusion model from scratch on the training site and evaluate its direct cross-site performance on the new site.
4.  Use `finetune.py` to apply the pre-trained weights to the new site and fine-tune the model using varying reference data budgets (1% to 100%).

## Acknowledgments

We acknowledge the open code provided by [PointNeXt](https://github.com/kentechx/pointnext/blob/main/pointnext/pointnext.py) for the 3D point cloud backbone and [U-Net](https://github.com/jaxony/unet-pytorch) for the optical imagery backbone.

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
