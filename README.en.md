# PyTorch Regression

[Русский](README.md) · **English**

A small PyTorch project that predicts a continuous value from ten numeric features. It covers tensor preparation, batching, training, validation, and saving and loading model weights.

## Data

The built-in scikit-learn Diabetes dataset has **442 examples and 10 features**. The target measures disease progression one year after baseline. The loader supplies centered and scaled features and requires no separate CSV. [Dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

## Model and training

```text
10 features → Linear(10, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1)
```

The network has **2,817 trainable parameters** and an unrestricted linear output.

| Setting | Value |
|---|---|
| Split | 354 train, 88 validation, split seed 42 |
| Training | 100 epochs, batch size 16 |
| Loss | MSE |
| Optimizer | AdamW, learning rate 0.001 |
| Device | CPU |

`train.py` uses `TensorDataset` and `random_split`. Every ten epochs it prints loss for the last batch and the mean across batches. It evaluates validation after the final epoch and saves `regression_model.pt`. `predict.py` loads the weights and prints the prediction and target for the first dataset row.

## Run on Windows

Verified with Python 3.13.2, PyTorch 2.15.0+cpu, and scikit-learn 1.9.1. Run from the repository root.

```powershell
git clone https://github.com/Awuiga/pytorch-regression.git
cd pytorch-regression
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python predict.py
```

Training starts from scratch and writes the local weights required for prediction. Use trusted checkpoint files.

## Reproducibility and limits

Seed 42 fixes the split. Initialization and batch ordering remain random. The script averages batch means, giving the smaller last batch equal weight. Validation runs once and there is no separate test split. The first demonstration row may be in train. The built-in scaling is applied before the split. A stricter experiment would fit scaling on train alone. The dataset and model are used here for a regression exercise. The full training and prediction check is recorded separately. [Verification](VERIFICATION.md)
