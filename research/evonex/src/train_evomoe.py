"""
EvoNex Training Pipeline
Trains the EvoMoE model on TESS light curve data.

Usage:
    python train_evomoe.py --config configs/default.yaml
    python train_evomoe.py --config configs/small.yaml --epochs 10
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from dataset import TESSDataset
from model_evonex import EvoMoE_Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("evonex.train")

import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("experiment_")]
    next_id = len(existing) + 1
    exp_dir = os.path.join(base_dir, f"experiment_{next_id:03d}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir, f"experiment_{next_id:03d}"


def train(config, config_path=None):
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    if train_cfg.get("seed"):
        torch.manual_seed(train_cfg["seed"])

    exp_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_cfg["experiment_dir"])
    exp_dir, exp_id = create_experiment_dir(exp_base)
    logger.info(f"Experiment directory: {exp_dir}")

    if config_path:
        shutil.copy2(config_path, os.path.join(exp_dir, "config.yaml"))

    dataset_version = data_cfg.get("version", "v1-curated")
    try:
        dataset = TESSDataset(dataset_version=dataset_version)
    except FileNotFoundError:
        logger.error(f"Dataset version '{dataset_version}' not found. Run build_dataset.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)

    model = EvoMoE_Model(
        lc_seq_length=model_cfg["lc_seq_length"],
        num_tic_features=model_cfg["num_tic_features"],
        embed_dim=model_cfg["embed_dim"],
        num_classes=model_cfg["num_classes"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    with open(os.path.join(exp_dir, "model_summary.txt"), "w") as f:
        f.write(f"Architecture: EvoMoE\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Input LC: (batch, {model_cfg['lc_seq_length']})\n")
        f.write(f"Input TIC: (batch, {model_cfg['num_tic_features']})\n")
        f.write(f"Output: (batch, {model_cfg['num_classes']})\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    metrics_log = []

    logger.info(f"Training for {train_cfg['epochs']} epochs...")

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for lc_tensor, tic_tensor, label_tensor in dataloader:
            optimizer.zero_grad()
            logits, weights = model(lc_tensor, tic_tensor)
            loss = criterion(logits, label_tensor)
            loss.backward()

            if train_cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])

            optimizer.step()
            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - start_time
        avg_loss = running_loss / max(len(dataloader), 1)

        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "time_seconds": round(epoch_time, 2),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        metrics_log.append(epoch_metrics)

        logger.info(f"Epoch [{epoch+1}/{train_cfg['epochs']}] Loss: {avg_loss:.4f} Time: {epoch_time:.2f}s")

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_cfg["weights_dir"])
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, "evomoe_weights.pth")
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Weights saved to {weights_path}")

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    import csv
    with open(os.path.join(exp_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "time_seconds", "lr"])
        writer.writeheader()
        writer.writerows(metrics_log)

    with open(os.path.join(exp_dir, "metrics.md"), "w") as f:
        f.write("# Training Metrics\n\n")
        f.write("| Epoch | Loss | Time (s) | LR |\n")
        f.write("|---|---|---|---|\n")
        for m in metrics_log:
            f.write(f"| {m['epoch']} | {m['loss']:.6f} | {m['time_seconds']:.2f} | {m['lr']:.8f} |\n")

    reproducibility_manifest = {
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "random_seed": train_cfg.get("seed", "None"),
        "dataset_version": dataset_version,
        "model_architecture": "EvoMoE"
    }
    with open(os.path.join(exp_dir, "reproducibility.json"), "w") as f:
        json.dump(reproducibility_manifest, f, indent=2)

    index_path = os.path.join(exp_base, "index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
    else:
        index = []

    index.append({
        "id": exp_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": config_path or "inline",
        "dataset": dataset_version,
        "final_loss": metrics_log[-1]["loss"] if metrics_log else None,
        "epochs": train_cfg["epochs"],
        "status": "completed",
    })

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Experiment {exp_id} logged to {index_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EvoNex EvoMoE Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.config)
    config = load_config(config_path)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    train(config, config_path=config_path)
