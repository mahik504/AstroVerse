import os
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, brier_score_loss, log_loss
import xgboost as xgb

from dataset import TESSDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("evonex.baselines")

# ==========================================
# 1. CNN Baseline
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, seq_length=2000, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        
        # Calculate flattened dimension
        x = torch.zeros(1, 1, seq_length)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        flat_dim = x.numel()
        
        self.fc = nn.Linear(flat_dim, num_classes)
        
    def forward(self, lc, tic_features):
        x = lc.unsqueeze(1) # (B, 1, L)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 2. ResNet Baseline
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self, seq_length=2000, num_classes=2):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 7, stride=2, padding=3)
        self.relu = nn.ReLU()
        self.block1 = ResBlock(16, 32)
        self.block2 = ResBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, lc, tic_features):
        x = lc.unsqueeze(1)
        x = self.relu(self.conv(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ==========================================
# 3. Time-Series Transformer Baseline
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TSTransformer(nn.Module):
    def __init__(self, seq_length=2000, num_classes=2, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, lc, tic_features):
        x = lc.unsqueeze(-1) # (B, L, 1)
        x = self.input_proj(x) # (B, L, d_model)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # add pos encoding
        x = self.transformer(x)
        # Global average pooling
        x = x.mean(dim=1)
        return self.fc(x)

# ==========================================
# 4. XGBoost Baseline
# ==========================================
def train_evaluate_xgboost(dataloader, seed=42):
    # Extract naive features from light curves (min, max, std, mean) + TIC
    X_train, y_train = [], []
    for lc, tic, labels in dataloader:
        lc_np = lc.numpy()
        tic_np = tic.numpy()
        
        # Simple feature engineering: depth, variance, mean
        lc_min = np.min(lc_np, axis=1, keepdims=True)
        lc_std = np.std(lc_np, axis=1, keepdims=True)
        
        features = np.hstack([lc_min, lc_std, tic_np])
        X_train.extend(features)
        y_train.extend(labels.numpy())
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    if len(np.unique(y_train)) < 2:
        return {"f1": 0.0, "auprc": 0.0, "ece": 0.0, "brier": 0.0, "nll": 0.0}

    clf = xgb.XGBClassifier(n_estimators=50, random_state=seed, use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_train)[:, 1]
    preds = clf.predict(X_train)
    
    f1 = f1_score(y_train, preds)
    auprc = average_precision_score(y_train, probs)
    brier = brier_score_loss(y_train, probs)
    ece = calculate_ece(probs, y_train)
    nll = log_loss(y_train, probs)
    
    return {"f1": float(f1), "auprc": float(auprc), "ece": float(ece), "brier": float(brier), "nll": float(nll)}

# ==========================================
# 5. Classical BLS SNR Baseline
# ==========================================
def evaluate_classical(dataloader):
    # We mock SNR as proportional to the maximum dip depth (min flux) since we don't have explicit SNR saved.
    all_probs = []
    all_labels = []
    
    for lc, tic, labels in dataloader:
        lc_np = lc.numpy()
        # Max dip (negative) -> pseudo SNR. Invert so deeper dip = higher score
        min_flux = np.min(lc_np, axis=1) 
        # Normalize to 0-1 for probabilities
        pseudo_snr = np.abs(min_flux)
        max_val = np.max(pseudo_snr) if np.max(pseudo_snr) > 0 else 1.0
        probs = np.clip(pseudo_snr / max_val, 0, 1)
        
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    
    if len(np.unique(all_labels)) < 2:
        return {"f1": 0.0, "auprc": 0.0, "ece": 0.0, "brier": 0.0, "nll": 0.0}

    f1 = f1_score(all_labels, preds)
    auprc = average_precision_score(all_labels, all_probs)
    brier = brier_score_loss(all_labels, all_probs)
    ece = calculate_ece(all_probs, all_labels)
    nll = log_loss(all_labels, all_probs)
    
    return {"f1": float(f1), "auprc": float(auprc), "ece": float(ece), "brier": float(brier), "nll": float(nll)}

# ==========================================
# Evaluator function
# ==========================================
def calculate_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(probs > bin_lower, probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def train_and_evaluate(model_name, model, dataloader, epochs=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        for lc, tic, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(lc, tic)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for lc, tic, labels in dataloader:
            outputs = model(lc, tic)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Prevent NaNs from propagating to sklearn
            outputs = torch.nan_to_num(outputs, nan=0.0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
            preds = torch.argmax(outputs, dim=1).numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Handle single class batch edge case in small demo datasets
    if len(np.unique(all_labels)) < 2:
        logger.warning("Only one class present in evaluation. Returning 0 for metrics.")
        return {"f1": 0.0, "auprc": 0.0, "ece": 0.0, "brier": 0.0, "nll": 0.0}

    f1 = f1_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_probs)
    brier = brier_score_loss(all_labels, all_probs)
    ece = calculate_ece(all_probs, all_labels)
    nll = log_loss(all_labels, all_probs)
    
    return {
        "f1": float(f1),
        "auprc": float(auprc),
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll)
    }

def run_baselines_with_seeds(version, num_seeds=3):
    logger.info(f"Running baselines on dataset version: {version} across {num_seeds} seeds")
    dataset = TESSDataset(dataset_version=version)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    seeds = [42, 43, 44, 45, 46][:num_seeds]
    results = {"CNN": [], "ResNet": [], "Transformer": [], "XGBoost": [], "Classical": []}
    
    for seed in seeds:
        logger.info(f"--- Running Seed: {seed} ---")
        
        metrics_cnn = train_and_evaluate("CNN", SimpleCNN(), dataloader, seed=seed)
        results["CNN"].append(metrics_cnn)
        
        metrics_resnet = train_and_evaluate("ResNet", ResNet1D(), dataloader, seed=seed)
        results["ResNet"].append(metrics_resnet)
        
        metrics_tf = train_and_evaluate("Transformer", TSTransformer(), dataloader, seed=seed)
        results["Transformer"].append(metrics_tf)
        
        metrics_xgb = train_evaluate_xgboost(dataloader, seed=seed)
        results["XGBoost"].append(metrics_xgb)
        
        # Classical is deterministic, but we'll append to keep structure
        metrics_classical = evaluate_classical(dataloader)
        results["Classical"].append(metrics_classical)
        
    # Compute Mean and Std
    final_report = {"dataset": version, "models": {}}
    for model_name, runs in results.items():
        keys = runs[0].keys()
        agg = {}
        for k in keys:
            vals = [run[k] for run in runs]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        final_report["models"][model_name] = agg
        logger.info(f"Final {model_name} Metrics: {agg}")

    output_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "baselines_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    logger.info(f"Baselines report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1-curated-100")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds for statistical evaluation")
    args = parser.parse_args()
    
    run_baselines_with_seeds(args.version, num_seeds=args.seeds)
