import json
import os
import argparse

def create_model_card_template(output_dir, model_name):
    card = {
        "model_name": model_name,
        "training_time_seconds": None,
        "parameter_count": None,
        "memory_mb": None,
        "flops_measured": None,
        "best_config": {},
        "dataset_version": None,
        "seed": None,
        "hardware_logged": {
            "cpu": None,
            "gpu": None,
            "cuda_version": None,
            "ram_gb": None,
            "os": None
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    card_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_model_card.json")
    with open(card_path, 'w') as f:
        json.dump(card, f, indent=4)
        
    print(f"Generated {card_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../experiments/baselines")
    args = parser.parse_args()
    
    baselines = [
        "Classical_BLS",
        "Statistical_LogisticRegression",
        "Tree_XGBoost",
        "DL_ResNet1D",
        "DL_Transformer"
    ]
    
    for b in baselines:
        create_model_card_template(args.dir, b)
