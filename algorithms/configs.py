import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BASE_LR = 1e-4
FP4_LR = 5e-4
PRECISIONS = ["fp32", "fp16", "fp8", "fp4"]
SEEDS = [42]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

configs = [
    {'precision': 'fp32', 'lr': 1e-4, 'name': 'FP32'},
    {'precision': 'fp16', 'lr': 1e-4, 'name': 'FP16'},
    {'precision': 'fp8', 'lr': 1e-4, 'name': 'FP8 (lr=1e-4)'},
    {'precision': 'fp4', 'lr': 1e-4, 'name': 'FP4 (lr=1e-4)'},
    {'precision': 'fp4', 'lr': 5e-4, 'name': 'FP4 (lr=5e-4)'}
]