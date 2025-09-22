# finetune_trainer.py
import os, yaml, argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# --------------------------
# IO helpers
# --------------------------
def _save_config(cfg, save_dir, filename="config.yaml"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"[finetune] saved config to {path}")

def _load_backbone_weights(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing or unexpected:
        print(f"[finetune] skipped {len(unexpected)} incompatible tensors:")
        for n in unexpected: print("  •", n)
    return model

# --------------------------
# Freeze / unfreeze modes
# --------------------------
S2_EARLY = ["s2_model.encoder.stem", "s2_model.encoder.residual_block1"]
S2_MID   = ["s2_model.encoder.residual_block2", "s2_model.encoder.residual_block3"]
S2_LATE  = ["s2_model.encoder.residual_block4", "s2_model.decoder"]
S2_HEAD  = ["s2_model.classifier"]

PC_EARLY = ["pc_model.encoder.encoder.stem", "pc_model.encoder.encoder.encoder.0"]
PC_MID   = ["pc_model.encoder.encoder.encoder.1", "pc_model.encoder.encoder.encoder.2"]
PC_LATE  = ["pc_model.encoder.encoder.encoder.3", "pc_model.encoder.backbone.head"]
PC_HEAD  = ["pc_model.decoder.cls_head"]

FUSE_HEAD = ["fuse_head"]
MF_BLOCK  = ["mf_module"]  # only train if use_ms==True

def _starts(n, prefixes): return any(n.startswith(p) for p in prefixes)
def _set_requires_grad(model, train_prefixes):
    for n, p in model.named_parameters():
        p.requires_grad = _starts(n, train_prefixes)

def mode_linear_probe(model, use_ms_fusion=True):
    train = S2_HEAD + PC_HEAD + FUSE_HEAD
    _set_requires_grad(model, train)

def mode_full_ft(model, use_ms_fusion=True):
    for _, p in model.named_parameters(): p.requires_grad = True
    if not use_ms_fusion:
        for n, p in model.named_parameters():
            if _starts(n, MF_BLOCK): p.requires_grad = False

def mode_freeze_last_k(model, k=1, use_ms_fusion=True):
    train = []
    if k >= 1: train += S2_LATE + PC_LATE
    if k >= 2: train += S2_MID  + PC_MID
    if k >= 3: train += S2_EARLY + PC_EARLY + MF_BLOCK
    train += S2_HEAD + PC_HEAD + FUSE_HEAD
    _set_requires_grad(model, train)

# --------------------------
# Adapters (1x1 residual)
# --------------------------
class Conv2dAdapter(nn.Module):
    def __init__(self, in_ch, bottleneck=16):
        super().__init__()
        self.down = nn.Conv2d(in_ch, bottleneck, 1, bias=False)
        self.act  = nn.ReLU(inplace=True)
        self.up   = nn.Conv2d(bottleneck, in_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.down.weight); nn.init.zeros_(self.up.weight)
    def forward(self, x): return x + self.up(self.act(self.down(x)))

class Conv1dAdapter(nn.Module):
    def __init__(self, in_ch, bottleneck=16):
        super().__init__()
        self.down = nn.Conv1d(in_ch, bottleneck, 1, bias=False)
        self.act  = nn.ReLU(inplace=True)
        self.up   = nn.Conv1d(bottleneck, in_ch, 1, bias=False)
        nn.init.kaiming_normal_(self.down.weight); nn.init.zeros_(self.up.weight)
    def forward(self, x): return x + self.up(self.act(self.down(x)))

def attach_adapters(model, s2_channels: int, pc_channels: int,
                    s2_bottleneck=16, pc_bottleneck=16):
    # S2: wrap first conv block in classifier with adapter
    if hasattr(model.s2_model, "classifier") and hasattr(model.s2_model.classifier, "convs"):
        first = model.s2_model.classifier.convs[0]
        model.s2_model.classifier.convs[0] = nn.Sequential(Conv2dAdapter(s2_channels, s2_bottleneck), first)
    # PC: insert adapter before cls_head
    if hasattr(model.pc_model, "decoder") and hasattr(model.pc_model.decoder, "cls_head"):
        model.pc_model.decoder.cls_head = nn.Sequential(Conv1dAdapter(pc_channels, pc_bottleneck),
                                                        model.pc_model.decoder.cls_head)

def mode_adapters(model, s2_channels=64, pc_channels=768, use_ms_fusion=True,
                    s2_bottleneck=16, pc_bottleneck=16):
    attach_adapters(model, s2_channels, pc_channels, s2_bottleneck, pc_bottleneck)
    for _, p in model.named_parameters(): p.requires_grad = False
    train = S2_HEAD + PC_HEAD + FUSE_HEAD
    if use_ms_fusion: train += MF_BLOCK
    _set_requires_grad(model, train)

# --------------------------
# L2-SP (optional)
# --------------------------
def snapshot_source_state(model):
    model._src_state = {k: v.detach().clone().to("cpu") for k, v in model.state_dict().items()}
    model._src_param_keys = set(dict(model.named_parameters()).keys()) & set(model._src_state.keys())

def l2sp_loss(model, alpha=5e-5):
    try: alpha = float(alpha)
    except Exception: alpha = 5e-5
    dev  = next(model.parameters()).device
    dtype= next(model.parameters()).dtype
    reg = torch.zeros((), device=dev, dtype=dtype)
    if not hasattr(model, "_src_state") or not hasattr(model, "_src_param_keys"): return reg
    for k, p in model.named_parameters():
        if not p.requires_grad: continue
        if k not in model._src_param_keys: continue
        src = model._src_state[k]
        if src.shape != p.shape: continue
        reg = reg + (p - src.to(device=dev, dtype=p.dtype)).pow(2).sum()
    return reg * torch.tensor(alpha, device=dev, dtype=dtype)

def monkeypatch_l2sp(model, alpha):
    if alpha is None or alpha == 0: return
    orig_training_step = model.training_step
    def training_step_with_l2sp(*args, **kwargs):
        loss = orig_training_step(*args, **kwargs)
        if isinstance(loss, torch.Tensor):
            return loss + l2sp_loss(model, alpha=alpha)
        if isinstance(loss, dict) and "loss" in loss:
            loss["loss"] = loss["loss"] + l2sp_loss(model, alpha=alpha)
            return loss
        return loss
    model.training_step = training_step_with_l2sp

# --------------------------
# Optimizer & Scheduler
# --------------------------
def make_optimizer_and_scheduler(model, cfg):
    params = filter(lambda p: p.requires_grad, model.parameters())
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 5e-4))
    opt_name = str(cfg.get("optimizer", "adamW")).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd)
    """
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    max_epochs = int(cfg["max_epochs"])
    warmup_epochs = max(1, int(0.1 * max_epochs))
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    
    return optimizer, {"scheduler": scheduler, "interval": "epoch"}
    """
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, 10)
    return optimizer, scheduler

# --------------------------
# Main train
# --------------------------
def train(cfg, ft_mode_cli=None):
    seed_everything(int(cfg.get("seed", 123)))

    # --- dirs & save cfg ---
    log_name = cfg["log_name"] if "log_name" in cfg else "finetune_run"
    save_dir = os.path.join(cfg.get("save_dir", "./runs"), log_name)
    log_dir  = os.path.join(save_dir, "wandblogs")
    chk_dir  = os.path.join(save_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True); os.makedirs(chk_dir, exist_ok=True)
    _save_config(cfg, log_dir)

    # --- logger & callbacks ---
    wandb_logger = WandbLogger(project=cfg.get("wandb_project", "TL-TSC"),
                                group=cfg.get("wandb_group", "finetune"),
                                save_dir=log_dir)
    metric = cfg.get("monitor_metric", "val_r2")
    early = EarlyStopping(monitor=metric, patience=int(cfg.get("patience", 15)), mode="max", verbose=True)
    ckpt  = ModelCheckpoint(monitor=metric, filename="best", dirpath=chk_dir, save_top_k=1, mode="max")
    callbacks = [early, ckpt]

    # --- datamodule (your original routing) ---
    print("[finetune] init dataset")
    from dataset.superpixel import SuperpixelDataModule
    data_module = SuperpixelDataModule(cfg)

    # --- model (same mapping as your trainer) ---
    task = cfg["task"]
    if task == "tsc_aligned":
        from model.decison_fuse_aligned import FusionModel
    elif task == "tsc":
        from model.decison_fuse import FusionModel

    model = FusionModel(cfg, n_classes=cfg["n_classes"])

    # heads fresh for B; then load A-weights (backbone-only)
    # (keep even when training from scratch; it just re-inits heads cleanly)
    if hasattr(model, "s2_model") or hasattr(model, "pc_model"):
        # If your class already does this, you can comment out:
        # Reinit S2 conv_seg, PC last Linear, and fusion head
        if hasattr(model, "s2_model") and hasattr(model.s2_model, "classifier") and hasattr(model.s2_model.classifier, "conv_seg"):
            m = model.s2_model.classifier.conv_seg
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight); 
                if m.bias is not None: nn.init.zeros_(m.bias)
        if hasattr(model, "pc_model") and hasattr(model.pc_model, "decoder") and hasattr(model.pc_model.decoder, "cls_head"):
            last = model.pc_model.decoder.cls_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.kaiming_normal_(last.weight); 
                if last.bias is not None: nn.init.zeros_(last.bias)
        if hasattr(model, "fuse_head") and hasattr(model.fuse_head, "reset_parameters"):
            model.fuse_head.reset_parameters()

    if str(cfg.get("pretrained_ckpt", "None")) != "None":
        model = _load_backbone_weights(model, cfg["pretrained_ckpt"])
        snapshot_source_state(model)  # for optional L2-SP

    # --- choose finetune mode ---
    ft_mode = (ft_mode_cli or cfg.get("ft_mode", "linear_probe")).lower()
    
    use_ms  = bool(cfg.get("use_ms", False))  # MF block in front of S2

    if ft_mode == "linear_probe":
        mode_linear_probe(model, use_ms_fusion=use_ms)
    elif ft_mode == "full_ft":
        mode_full_ft(model, use_ms_fusion=use_ms)
    elif ft_mode == "adapter":
        # your channel counts
        s2_ch = int(cfg.get("s2_head_in_ch", 1024))
        pc_ch = int(cfg.get("pc_head_in_ch", 768))
        mode_adapters(model, s2_channels=s2_ch, pc_channels=pc_ch, use_ms_fusion=use_ms,
                        s2_bottleneck=int(cfg.get("s2_adapter_bottleneck", 16)),
                        pc_bottleneck=int(cfg.get("pc_adapter_bottleneck", 16)))
    else:
        k_last=int(ft_mode.split('_')[-1])
        mode_freeze_last_k(model, k=k_last, use_ms_fusion=use_ms)

    # --- optional L2-SP (set l2sp_alpha in YAML; works with any mode) ---
    monkeypatch_l2sp(model, cfg.get("l2sp_alpha", None))

    # --- optimizer & scheduler (single LR on trainables) ---
    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg)
    model.configure_optimizers = lambda: {"optimizer": optimizer, "lr_scheduler": scheduler}

    # --- trainer ---
    trainer = Trainer(
        max_epochs=int(cfg["max_epochs"]),
        logger=[wandb_logger],
        callbacks=callbacks,
        devices=int(cfg.get("gpus", 1)),
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    # --- go ---
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to YAML or JSON config")
    ap.add_argument("--ft_mode", default=None, help="Override: linear_probe | freeze_last_k | full_ft | adapters")
    ap.add_argument("--task", default=None, help="Override: tsc | tsc_align")
    ap.add_argument("--pretrained_ckpt", default=None, help="Override: linear_probe | freeze_last_k | full_ft | adapters")
    ap.add_argument("--log_name", default=None)
    args = ap.parse_args()
    with open(args.cfg, "r") as f:
        cfg = json.load(f) if args.cfg.endswith(".json") else yaml.safe_load(f)
    # sensible defaults if not present
    cfg.setdefault("save_dir", "tl_logs")
    cfg.setdefault("log_name", "finetune")
    cfg.setdefault("optimizer", "adamW")
    cfg.setdefault("lr", 1e-3)
    cfg.setdefault("max_epochs", 100)
    cfg.setdefault("patience", 15)
    cfg.setdefault("s2_head_in_ch", 1024)
    cfg.setdefault("pc_head_in_ch", 768)
    cfg.setdefault("task", args.task)
    cfg.setdefault("pretrained_ckpt", args.pretrained_ckpt)
    
    # Explicit overrides from CLI if provided
    if args.task is not None:
        cfg["task"] = args.task

    if args.pretrained_ckpt is not None:
        cfg["pretrained_ckpt"] = args.pretrained_ckpt
    
    if args.log_name is not None:
        cfg["log_name"] = args.log_name

    if args.ft_mode is not None:
        cfg["ft_mode"] = args.ft_mode
    train(cfg, ft_mode_cli=args.ft_mode)
