import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

# from pytorch_lightning.utilities.model_summary import ModelSummary
from dataset.balanced_dataset import BalancedDataModule
from model.fuse import FusionModel


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
    data_module = BalancedDataModule(config)

    # Use the calculated input channels from the DataModule to initialize the model
    if config["pretrained_ckpt"] != "None":
        # load backbone weights only, ignore head mismatch
        model = FusionModel.load_from_checkpoint(
            config["pretrained_ckpt"],
            n_classes=config["n_classes"],
            strict=False, # only matching parameters (the backbone) get loaded;
            **config
        )
        # freeze encoder
        # for p in model.s2_model.parameters(): p.requires_grad = False
        # for p in model.pc_model.parameters(): p.requires_grad = False
    else:
        model = FusionModel(config, n_classes=config["n_classes"],)
    # print(ModelSummary(model, max_depth=-1))  # Prints the full model summary

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=config["max_epochs"],
        logger=[wandb_logger],
        callbacks=early_stopping,  # [early_stopping, checkpoint_callback],
        devices=config["gpus"],
        num_nodes=1,
        strategy="ddp"
    )
    # Train the model
    trainer.fit(model, data_module)

    # using a pandas DataFrame to recode best results

    # Test the model after training
    trainer.test(model, data_module)
