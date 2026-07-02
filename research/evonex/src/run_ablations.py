import os
import json
import logging
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from thop import profile

from dataset import TESSDataset
from model_evonex import EvoMoE_Model
from run_baselines import train_and_evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evonex.ablations")

def profile_model(model, lc_shape=(1, 2000), tic_shape=(1, 12)):
    model.eval()
    dummy_lc = torch.randn(*lc_shape)
    dummy_tic = torch.randn(*tic_shape)
    
    # 1. Parameter Count
    params = sum(p.numel() for p in model.parameters())
    
    # 2. FLOPs
    try:
        flops, _ = profile(model, inputs=(dummy_lc, dummy_tic), verbose=False)
    except Exception as e:
        logger.warning(f"FLOP profiling failed: {e}")
        flops = 0
        
    # 3. Inference Time
    starts = []
    ends = []
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            model(dummy_lc, dummy_tic)
        # Measure
        for _ in range(50):
            starts.append(time.time())
            model(dummy_lc, dummy_tic)
            ends.append(time.time())
            
    inf_time_ms = (np.mean(ends) - np.mean(starts)) * 1000
    
    # 4. GPU Memory
    gpu_mem = 0
    if torch.cuda.is_available():
        model.cuda()
        dummy_lc = dummy_lc.cuda()
        dummy_tic = dummy_tic.cuda()
        torch.cuda.reset_peak_memory_stats()
        model(dummy_lc, dummy_tic)
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
        model.cpu()
        
    return {
        "parameters": params,
        "flops": flops,
        "inference_ms": float(inf_time_ms),
        "gpu_memory_mb": float(gpu_mem)
    }

def run_ablations(version, num_seeds=3):
    logger.info(f"Running ablations on dataset {version} over {num_seeds} seeds")
    dataset = TESSDataset(dataset_version=version)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    seeds = [42, 43, 44, 45, 46][:num_seeds]
    
    ablation_configs = {
        "CNN_Only": ["cnn"],
        "Transformer_Only": ["transformer"],
        "Physics_Only": ["physics"],
        "CNN_Transformer": ["cnn", "transformer"],
        "Full_EvoMoE": ["cnn", "transformer", "physics"]
    }
    
    results = {k: [] for k in ablation_configs.keys()}
    profiles = {}
    
    for name, experts in ablation_configs.items():
        logger.info(f"--- Ablation: {name} (Experts: {experts}) ---")
        
        # Profile once
        model = EvoMoE_Model(active_experts=experts)
        prof = profile_model(model)
        profiles[name] = prof
        logger.info(f"Profile: {prof}")
        
        for seed in seeds:
            model = EvoMoE_Model(active_experts=experts)
            metrics = train_and_evaluate(name, model, dataloader, epochs=5, seed=seed)
            results[name].append(metrics)
            
    # Compute $\Delta$ (Delta) against Full EvoMoE
    final_report = {"dataset": version, "models": {}}
    
    for name, runs in results.items():
        keys = runs[0].keys()
        agg = {}
        for k in keys:
            vals = [run[k] for run in runs]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        
        # Add profile info
        agg.update(profiles[name])
        final_report["models"][name] = agg
        
    # Calculate Deltas
    full_metrics = final_report["models"]["Full_EvoMoE"]
    for name in final_report["models"]:
        if name != "Full_EvoMoE":
            agg = final_report["models"][name]
            agg["delta_f1"] = agg["f1_mean"] - full_metrics["f1_mean"]
            agg["delta_auprc"] = agg["auprc_mean"] - full_metrics["auprc_mean"]
            agg["delta_ece"] = agg["ece_mean"] - full_metrics["ece_mean"]
            agg["delta_nll"] = agg["nll_mean"] - full_metrics["nll_mean"]
            
    output_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "ablations_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    logger.info(f"Ablation report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1-curated-100")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    
    run_ablations(args.version, args.seeds)
