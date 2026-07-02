# Experiments Directory

Each training run creates a numbered subdirectory:

```
experiments/
├── experiment_001/
│   ├── config.yaml       # Frozen copy of training config
│   ├── metrics.json       # Epoch-by-epoch metrics
│   ├── metrics.csv        # Same metrics in CSV format
│   ├── model_summary.txt  # Architecture + parameter count
│   └── README.md          # Run notes
├── experiment_002/
│   └── ...
└── index.json             # Registry of all experiments
```

## Experiment Registry

The `index.json` file tracks all runs:

```json
[
  {
    "id": "experiment_001",
    "date": "2026-07-02",
    "config": "configs/small.yaml",
    "dataset": "4 curated TIC targets",
    "best_f1": null,
    "status": "completed",
    "notes": "Smoke test run"
  }
]
```

## Adding an Experiment

Training scripts automatically create experiment directories. To manually log:

1. Copy config to `experiment_NNN/config.yaml`
2. Save metrics to `experiment_NNN/metrics.json`
3. Update `index.json`
