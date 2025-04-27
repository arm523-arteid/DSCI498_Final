import pandas as pd
import numpy as np
import requests
import io

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import subprocess

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    subprocess.run(["python", script_name], check=True)

def main():
    # Run each script in order
    run_script("process_data.py")
    run_script("nn.py")
    run_script("vae_train.py")
    run_script("vae_train_augment.py")
    run_script("nn_combined.py")

if __name__ == "__main__":
    main()