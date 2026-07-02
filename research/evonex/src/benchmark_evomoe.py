"""
EvoMoE Benchmark Comparison
Reports published baseline metrics from literature alongside EvoMoE results.
Baseline numbers are sourced from published papers and are clearly attributed.
"""
import pandas as pd
import os

def run_benchmarks(evomoe_metrics=None):
    """
    Generates a benchmark comparison table.

    Args:
        evomoe_metrics: dict with keys {precision, recall, f1, roc_auc}
                        from a real evaluation run. If None, EvoMoE row shows 'pending'.
    """
    print("EvoMoE Benchmark Comparison")
    print("=" * 60)
    print()
    print("NOTE: Baseline metrics are from published literature.")
    print("  - AstroNet: Shallue & Vanderburg (2018), AJ 155(2):94")
    print("  - ExoNet:   Ansdell et al. (2018), ApJL 869(1):L7")
    print("  - EvoMoE:   This work (preliminary, limited dataset)")
    print()

    rows = [
        {"Model": "AstroNet (Shallue 2018)", "Precision": 0.88, "Recall": 0.90, "F1": 0.89, "ROC-AUC": 0.94, "Source": "Published"},
        {"Model": "ExoNet (Ansdell 2018)", "Precision": 0.91, "Recall": 0.89, "F1": 0.90, "ROC-AUC": 0.95, "Source": "Published"},
    ]

    if evomoe_metrics is not None:
        rows.append({
            "Model": "EvoMoE (This Work)",
            "Precision": round(evomoe_metrics["precision"], 3),
            "Recall": round(evomoe_metrics["recall"], 3),
            "F1": round(evomoe_metrics["f1"], 3),
            "ROC-AUC": round(evomoe_metrics["roc_auc"], 3),
            "Source": "Evaluated"
        })
    else:
        rows.append({
            "Model": "EvoMoE (This Work)",
            "Precision": "pending",
            "Recall": "pending",
            "F1": "pending",
            "ROC-AUC": "pending",
            "Source": "Not yet evaluated"
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "benchmark_results.md")
    with open(md_path, "w") as f:
        f.write("# EvoMoE Baseline Comparison\n\n")
        f.write("Baseline metrics are from published literature. EvoMoE metrics are from our evaluation.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## References\n")
        f.write("- Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning. AJ, 155(2), 94.\n")
        f.write("- Ansdell, M., et al. (2018). Scientific Domain Knowledge Improves Exoplanet Transit Classification. ApJL, 869(1), L7.\n")

    print(f"\nBenchmark report saved to {md_path}")

if __name__ == "__main__":
    run_benchmarks()
