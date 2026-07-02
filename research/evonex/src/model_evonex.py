import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleCNN_Expert(nn.Module):
    """
    Expert 1: Multi-Scale CNN (Local Expert).
    Captures short-duration sharp dips (eclipsing binaries)
    and long-duration shallow dips (planets) simultaneously using different kernel sizes.
    """
    def __init__(self, seq_len=2000, embed_dim=128):
        super().__init__()

        self.conv1_small = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv1_med   = nn.Conv1d(1, 16, kernel_size=11, padding=5)
        self.conv1_large = nn.Conv1d(1, 16, kernel_size=21, padding=10)

        self.pool = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(48, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(4)

        self.fc = nn.Linear(64 * (seq_len // 16), embed_dim)

        self.conf_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x_s = F.relu(self.conv1_small(x))
        x_m = F.relu(self.conv1_med(x))
        x_l = F.relu(self.conv1_large(x))

        x_cat = torch.cat([x_s, x_m, x_l], dim=1)
        x_cat = self.pool(x_cat)

        x_cat = F.relu(self.conv2(x_cat))
        x_cat = self.pool2(x_cat)

        x_flat = x_cat.view(x_cat.size(0), -1)
        embedding = F.relu(self.fc(x_flat))

        confidence = self.conf_head(embedding)
        return embedding, confidence


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the Transformer.
    Without this, the Transformer cannot distinguish temporal order.
    """
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer_Expert(nn.Module):
    """
    Expert 2: Temporal Transformer (Global Expert).
    Captures complex periodicities and long-range dependencies in the light curve.
    Uses sinusoidal positional encoding to preserve temporal order.
    """
    def __init__(self, seq_len=2000, embed_dim=128, n_heads=4):
        super().__init__()

        self.patch_size = 20
        self.num_patches = seq_len // self.patch_size

        self.patch_proj = nn.Linear(self.patch_size, 32)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=32, max_len=self.num_patches)

        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=n_heads, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(self.num_patches * 32, embed_dim)
        self.conf_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)

        x_patched = x.view(batch_size, self.num_patches, self.patch_size)

        x_proj = self.patch_proj(x_patched)
        x_proj = self.pos_encoding(x_proj)
        x_trans = self.transformer(x_proj)

        x_flat = x_trans.view(batch_size, -1)
        embedding = F.relu(self.fc(x_flat))

        confidence = self.conf_head(embedding)
        return embedding, confidence


class PhysicsMLP_Expert(nn.Module):
    """
    Expert 3: Physics-Aware MLP (Stellar Expert).
    Provides the physical context of the star (Temp, Radius, Mass)
    to validate transit depth physics.
    Expects normalized stellar features.
    """
    def __init__(self, num_tic_features=12, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_tic_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim),
            nn.ReLU()
        )
        self.conf_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedding = self.net(x)
        confidence = self.conf_head(embedding)
        return embedding, confidence


class EvoMoE_Model(nn.Module):
    """
    EvoMoE: Evolutionary Mixture of Experts.
    A Physics-Aware Confidence-Guided Multimodal Framework for
    Robust Exoplanet Transit Detection.
    """
    def __init__(self, lc_seq_length=2000, num_tic_features=12, embed_dim=128, num_classes=2):
        super().__init__()

        self.cnn_expert = MultiScaleCNN_Expert(seq_len=lc_seq_length, embed_dim=embed_dim)
        self.transformer_expert = Transformer_Expert(seq_len=lc_seq_length, embed_dim=embed_dim)
        self.physics_expert = PhysicsMLP_Expert(num_tic_features=num_tic_features, embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, lc_tensor, tic_tensor):
        """
        lc_tensor: (batch, seq_len)
        tic_tensor: (batch, num_tic_features)
        Returns: logits, explainability_weights
        """
        lc_tensor = lc_tensor.unsqueeze(1)

        emb_cnn, conf_cnn = self.cnn_expert(lc_tensor)
        emb_trans, conf_trans = self.transformer_expert(lc_tensor)
        emb_phys, conf_phys = self.physics_expert(tic_tensor)

        confidences = torch.cat([conf_cnn, conf_trans, conf_phys], dim=1)

        gating_weights = F.softmax(confidences, dim=1)

        w_cnn = gating_weights[:, 0].unsqueeze(1)
        w_trans = gating_weights[:, 1].unsqueeze(1)
        w_phys = gating_weights[:, 2].unsqueeze(1)

        fused_embedding = (w_cnn * emb_cnn) + (w_trans * emb_trans) + (w_phys * emb_phys)

        logits = self.classifier(fused_embedding)

        return logits, gating_weights

if __name__ == "__main__":
    print("Testing EvoMoE (Evolutionary Mixture of Experts) Architecture...")

    batch_size = 4
    sample_lc = torch.randn(batch_size, 2000)
    sample_tic = torch.randn(batch_size, 12)

    model = EvoMoE_Model()

    logits, expert_weights = model(sample_lc, sample_tic)

    print(f"\nInput Light Curve: {sample_lc.shape}")
    print(f"Input TIC Metadata: {sample_tic.shape}")
    print(f"Output Predictions: {logits.shape}")
    print(f"Output Expert Weights: {expert_weights.shape}")

    print("\n--- Explainability Output (Sample 0) ---")
    print(f"Prediction Logits: {logits[0].detach().numpy()}")
    print(f"CNN Expert Contribution:         {expert_weights[0][0].item()*100:.2f}%")
    print(f"Transformer Expert Contribution: {expert_weights[0][1].item()*100:.2f}%")
    print(f"Physics MLP Contribution:        {expert_weights[0][2].item()*100:.2f}%")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\nArchitecture validation: PASSED")
