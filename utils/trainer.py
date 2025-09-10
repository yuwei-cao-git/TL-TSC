import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import yaml
from model.finetune import reinit_classifier_heads, GradualUnfreeze

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
    metric = "val_r2"
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
    if config["task"] in ["tsc", "lsc", "tsc_mid", "tsc_aligned", "tsc_mid_decision"]:
        if config["dataset"] in ["ovf", "rmf"]:
            from dataset.balanced_dataset import BalancedDataModule
            data_module = BalancedDataModule(config)
        elif config["dataset"] == 'multi':
            from dataset.multidataset import MultiSourceDataModule
            data_module = MultiSourceDataModule(train_sources=("ovf","rmf"),  # train on both
                                                val_sources=("ovf",),         # validate on OVF only
                                                test_sources=("ovf",),        # test on OVF only
                                            )
        else:
            from dataset.superpixel import SuperpixelDataModule
            data_module = SuperpixelDataModule(config)
    elif config["task"] in ["pc_tsc", "pc_lsc"]:
        from dataset.pc import PcDataModule
        data_module = PcDataModule(config)
    else:
        from dataset.s2 import S2DataModule
        data_module = S2DataModule(config)
    
    # Use the calculated input channels from the DataModule to initialize the model
    if config["task"] == "tsc_mid":
        from model.fuse import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "tsc_mid_decision":
        from model.mh_df import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "tsc":
        from model.decison_fuse import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "tsc_aligned":
        if config["pretrained_ckpt"] != "None":
            from model.finetune import FusionModel
        else:    
            from model.decison_fuse_aligned import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "lsc":
        from model.lsc import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    elif config["task"] == "pc_tsc":
        from model.pc_model import PCModel
        model = PCModel(config, n_classes=config["n_classes"])
    elif config["task"] == "pc_lsc":
        from model.lsc_pc import PCModel
        model = PCModel(config, n_classes=config["n_classes"])
    elif config["task"] == "img_tsc":
        from model.s2_model import S2Model
        model = S2Model(config, n_classes=config["n_classes"])
    else:
        from model.top2 import FusionModel
        model = FusionModel(config, n_classes=config["n_classes"])
    
    #print(ModelSummary(model, max_depth=3))
    reinit_classifier_heads(model)
        
    if config["pretrained_ckpt"] != "None":
        # load backbone weights only, ignore head mismatch
        model = load_backbone_weights(model, config["pretrained_ckpt"])
        gradual = GradualUnfreeze(e1=config.get("unfreeze_e1", 3), e2=config.get("unfreeze_e2", 8))
        callbacks.append(gradual)

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=[wandb_logger],
        callbacks=callbacks,
        devices=config["gpus"],
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    
    # Train the model
    trainer.fit(model, data_module)

    # Test the model after training
    trainer.test(model, data_module)
