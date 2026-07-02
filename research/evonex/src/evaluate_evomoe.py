"""
EvoMoE Evaluation Script
Runs the model on test data and computes real sklearn metrics.
"""
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from model_evonex import EvoMoE_Model

def evaluate_model(weights_path=None, test_lc=None, test_tic=None, test_labels=None):
    """
    Evaluates EvoMoE on provided test data and computes real metrics.
    If no test data is provided, runs an architecture validation smoke test.
    """
    model = EvoMoE_Model(num_classes=2)

    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        print(f"Loaded weights from {weights_path}")
    else:
        print("WARNING: No weights provided. Running with untrained model (smoke test only).")

    model.eval()

    if test_lc is not None and test_tic is not None and test_labels is not None:
        with torch.no_grad():
            logits, gating_weights = model(test_lc, test_tic)
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()

        y_true = test_labels.numpy()
        y_pred = preds.numpy()
        y_prob = probs.numpy()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = float('nan')

        print("\n--- EvoMoE Evaluation Results ---")
        print(f"Samples Evaluated: {len(y_true)}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["False Positive", "Exoplanet"], zero_division=0))

        return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}
    else:
        print("\n--- Architecture Smoke Test (Untrained) ---")
        sample_lc = torch.randn(4, 2000)
        sample_tic = torch.randn(4, 13)
        logits, gating_weights = model(sample_lc, sample_tic)

        print(f"Input: lightcurve {sample_lc.shape}, stellar features {sample_tic.shape}")
        print(f"Output: logits {logits.shape}, gating weights {gating_weights.shape}")
        print(f"Gating distribution (sample 0): CNN={gating_weights[0][0]:.3f}, Transformer={gating_weights[0][1]:.3f}, Physics={gating_weights[0][2]:.3f}")
        print("Architecture validation: PASSED")
        return None

if __name__ == "__main__":
    evaluate_model()
