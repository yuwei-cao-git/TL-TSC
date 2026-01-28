import os
import torch
from torch import nn
from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
# from pytorch_lightning.utilities.model_summary import ModelSummary
import yaml

def load_backbone_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Get the actual state_dict (Lightning saves it under 'state_dict')
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 1. Filter out classifier layers with shape mismatches (e.g. output heads)
    model_state = model.state_dict()
    compatible_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }

    # 2. Load only compatible tensors
    missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)

    # 3. log what was skipped
    if missing or unexpected:
        print(f"Skipped {len(unexpected)} incompatible or filtered tensors:")
        for name in unexpected:
            print(f"  • {name}")

    return model

def save_config(cfg, save_dir, filename="config.yaml"):
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, filename)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Saved config to {config_path}")


def initialize_weights(m):
    # A recursive function to apply initialization to all relevant layers
    if isinstance(m, nn.Linear):
        # Kaiming/He initialization for linear layers followed by ReLU
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


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
    metric = "val_r2" if config["task"] in ["tsc", "tsc_mid", "tsca", "img_tsc", "pc_tsc"] else "val_f1"
    early_stopping = EarlyStopping(
        monitor=metric,  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="max",  # Set "min" for validation loss
        verbose=True,
    )

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,  # Track the validation loss
        filename="final_model",
        dirpath=chk_dir,
        save_top_k=1,  # Only save the best model
        mode="max",  # We want to minimize the validation loss
    )

    callbacks = [early_stopping, checkpoint_callback]

    print("start setting dataset")
    # Initialize the DataModule
    if config["task"] in ["tsc", "lsc", "tsc_mid", "tsca"]:
        if config["dataset"] in ["rmf_common", "wrf_common", "rmf_msp", "wrf_msp", "rmf_4class", "wrf_4class"]:
            from dataset.common import SuperpixelDataModule
            data_module = SuperpixelDataModule(config)
        else:
            from dataset.superpixel import SuperpixelDataModule
            data_module = SuperpixelDataModule(config)
    elif config["task"] == "pc_tsc":
        from dataset.pc import PcDataModule
        data_module = PcDataModule(config)
        from model.pc_model import PCModel
        model = PCModel(config, n_classes=config["n_classes"])
    else:
        from dataset.s2 import S2DataModule
        data_module = S2DataModule(config)
        from model.s2_model import S2Model
        model = S2Model(config, n_classes=config["n_classes"])

    # Use the calculated input channels from the DataModule to initialize the model
    if config["task"] == "tsc_mid":
        from model.fuse import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "tsc":
        from model.decison_fuse import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "tsca":
        from model.decision_fusion_aligned import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "lsc":
        from model.decison_fuse_aligned_lsc import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])

    if config["pretrained_ckpt"] != "None":
        # load backbone weights only, ignore head mismatch
        model = load_backbone_weights(model, config["pretrained_ckpt"])
    else:
        initialize_weights(model)

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=[wandb_logger],
        callbacks=callbacks,
        gradient_clip_val=0.5,
        # devices=config["gpus"],
        num_nodes=1,
        strategy="auto",  # DDPStrategy(find_unused_parameters=False)
    )

    if config["mode"] == "train":
        # Train the model
        trainer.fit(model, data_module)

        # Test the model after training
        trainer.test(model, data_module)
    else:
        trainer.test(model, data_module)