import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
# from pytorch_lightning.utilities.model_summary import ModelSummary
from model.fuse import FusionModel
import yaml

def load_backbone_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Get the actual state_dict (Lightning saves it under 'state_dict')
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Filter out classifier layers with shape mismatches (e.g. output heads)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not (
        "cls_head" in k or "fuse_head.mlp_block.fc3" in k
    )}

    # Load the rest of the weights
    model.load_state_dict(filtered_state_dict, strict=False)
    return model

def save_config(cfg, save_dir, filename="config.yaml"):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, filename)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Saved config to {config_path}")
    
def train(config):
    seed_everything(123)
    log_name = config["log_name"]
    save_dir = os.path.join(config["save_dir"], log_name)
    log_dir = os.path.join(save_dir, "wandblogs")
    chk_dir = os.path.join(save_dir, "checkpoints")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(chk_dir):
        os.mkdir(chk_dir)
    
    save_config(config, log_dir)

    # Initialize WandB, CSV Loggers
    wandb_logger = WandbLogger(
        project="TL-TSC",
        group="train_group",
        save_dir=log_dir,
        # log_model=True,
    )

    # Define a checkpoint callback to save the best model
    metric = "ave_val_r2"
    early_stopping = EarlyStopping(
        monitor=metric,  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="max",  # Set "min" for validation loss
        verbose=True,
    )
    print("start setting dataset")
    # Initialize the DataModule
    if config["dataset"] in ["rmf_superpixel_dataset", "ovf_superpixel_dataset"]:
        from dataset.superpixel import SuperpixelDataModule
        data_module = SuperpixelDataModule(config)
    else:
        from dataset.balanced_dataset import BalancedDataModule
        data_module = BalancedDataModule(config)
    # Use the calculated input channels from the DataModule to initialize the model
    
    model = FusionModel(config, n_classes=config["n_classes"])
    if config["pretrained_ckpt"] != "None":
        # load backbone weights only, ignore head mismatch
        model = load_backbone_weights(model, config["pretrained_ckpt"])

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=[wandb_logger],
        callbacks=early_stopping,  # [early_stopping, checkpoint_callback],
        devices=config["gpus"],
        num_nodes=1,
        strategy='ddp'# DDPStrategy #(find_unused_parameters=True)
    )
    
    # Train the model
    trainer.fit(model, data_module)

    # Test the model after training
    trainer.test(model, data_module)
