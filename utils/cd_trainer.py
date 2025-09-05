import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger

from dataset.combined import build_multi_region_dm
from model.mh_df import MultiHeadFusionModel

def train(config):
    seed_everything(config.get("seed", 123), workers=True)

    # Build datamodule from two YAML configs
    dm, cfg_A, cfg_B = build_multi_region_dm(
        config.get("cfg_path_A", "configs/config_rmf.yaml"),
        config.get("cfg_path_B", "configs/config_ovf_coarser.yaml"),
    )

    # Region-specific head sizes
    region_class_map = {"A": 9, "B": 6}

    # Merge base config (A first, B fills missing keys)
    base_cfg = {**cfg_A, **{k: v for k, v in cfg_B.items() if k not in cfg_A}}

    # Model
    model = MultiHeadFusionModel(
        config=base_cfg,
        region_class_map=region_class_map,
        feat_dim=base_cfg.get("fusion_out_ch", 256),
    )

    # --- Logging ---
    logger = WandbLogger(
        project=config.get("wandb_project", "tree-proportion"),
        name=config.get("run_name", None),
    )

    # --- Callbacks ---
    ckpt_cb = ModelCheckpoint(
        dirpath=config.get("save_dir", "./checkpoints"),
        filename="{epoch}-{valr2_mean:.4f}",
        monitor="val_epoch_r2/mean",
        mode="max",
        save_last=True,
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # --- Trainer strategy / devices ---
    gpus = int(base_cfg.get("gpus", 0))

    trainer = Trainer(
        max_epochs=base_cfg["max_epochs"],
        devices=gpus if gpus > 0 else None,
        strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=False),
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        gradient_clip_val=config.get("grad_clip", 1.0),
        log_every_n_steps=config.get("log_every_n_steps", 10),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        num_sanity_val_steps=2,
    )

    # Fit
    trainer.fit(model, datamodule=dm)

    # Test best checkpoint
    trainer.test(model=model, datamodule=dm, ckpt_path="best")