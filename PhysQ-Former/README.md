# PhysQ-Former

PhysQ-Former is a physically structured Transformer for QPF-oriented precipitation-intensity estimation from multilevel CMA-MESO atmospheric predictors.

The repository is organized as a clean, modular codebase for training and ablation experiments. Data files are not included.

## Feature layout

The model expects **87 atmospheric predictors** per sample:

```text
11 single-level predictors
+ 19 geopotential-height levels (Z)
+ 19 relative-humidity levels (RH)
+ 19 specific-humidity levels (q)
+ 19 temperature levels (T)
= 87 input features
```

The expected feature order is:

```text
[ surface_1 ... surface_11,
  Z_1000 ... Z_100,
  RH_1000 ... RH_100,
  q_1000 ... q_100,
  T_1000 ... T_100 ]
```

Each `train_data_*.pkl` file should use the following table structure:

```text
column 0: precipitation target in mm/hr
columns 1-87: atmospheric predictors in the order above
```

## Directory layout

```text
physqformer_github/
├── physqformer/
│   ├── __init__.py
│   ├── config.py          # paths, thresholds, feature layout, ablation settings
│   ├── data.py            # data loading, 87-dim validation, split, scaling, sampler
│   ├── models.py          # PhysQ-Former, vertical thermodynamic module, flat MLP baseline
│   ├── losses.py          # asymmetric regression loss and auxiliary occurrence loss
│   ├── evaluation.py      # RMSE/MAE/R2/NSE/KGE/PBIAS/CSI/POD/FAR metrics
│   ├── plotting.py        # diagnostic figures
│   ├── experiment.py      # training loop and ablation suite
│   └── utils.py
├── train.py               # command-line entry point
├── requirements.txt
├── .gitignore
├── data/                  # put train_data_1.pkl ... train_data_13.pkl here
├── checkpoints/           # saved models
└── outputs/               # figures and result csv files
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Place your data files in `data/`:

```text
data/train_data_1.pkl
...
data/train_data_12.pkl
data/train_data_13.pkl   # synthetic dry samples, training only
```

Run the full PhysQ-Former model:

```bash
python train.py --data-dir data --epochs 80 --batch-size 1024
```

Run all ablation experiments:

```bash
python train.py --data-dir data --epochs 80 --batch-size 1024 --ablation
```

Use file-based split instead of random rainy-sample split:

```bash
python train.py --data-dir data --split-mode by_file
```

## Notes

- Synthetic dry samples are used only for training regularization.
- Validation and test evaluation use real rainy samples only.
- The occurrence branch is an auxiliary branch by default; final rainy-sample inference uses the conditional-intensity output without occurrence-probability gating.
- The code checks the input feature dimension and raises an error if a file does not contain exactly 87 predictors.
