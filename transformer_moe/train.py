import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import time
import copy
from model import MixtureOfExperts


NUM_RUNS = 10
NUM_EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
BASE_SEED = 42

CSV_FILE_PATH = r'...\moe_test_1.csv'
BASE_OUTPUT_DIR = r'....\transformer_moe\result'

WEIGHTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'weights')

os.makedirs(WEIGHTS_DIR, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Data loaded successfully: {CSV_FILE_PATH}")
    except FileNotFoundError:
        exit(f"Error: File not found '{CSV_FILE_PATH}'")

    split_column = df.columns[0]

    df_train = df[df[split_column] == 'Train'].copy()
    df_val = df[(df[split_column] == 'Val') | (df[split_column] == 'Validation')].copy()

    if df_train.empty:
        exit("Error: No 'Train' data found after filtering.")
    if df_val.empty:
        exit("Error: No 'Val' or 'Validation' data found after filtering. Model selection cannot be performed.")

    print(f"Number of training set samples: {len(df_train)}")
    print(f"Number of val set samples: {len(df_val)}")

    feature_columns = df.columns[3:]
    X_train = torch.tensor(df_train[feature_columns].values, dtype=torch.float32)
    y_train = torch.tensor(df_train.iloc[:, 2].values, dtype=torch.float32).view(-1, 1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_val = torch.tensor(df_val[feature_columns].values, dtype=torch.float32)
    y_val = torch.tensor(df_val.iloc[:, 2].values, dtype=torch.float32).view(-1, 1)
    val_dataset = TensorDataset(X_val, y_val)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"\nWill perform {NUM_RUNS} independent training runs (Transformer MoE)...")

    for run in range(NUM_RUNS):
        current_seed = BASE_SEED + run
        seed_everything(current_seed)

        print(f"\n{'=' * 10}  Run {run + 1}/{NUM_RUNS} (Seed: {current_seed}) {'=' * 10}")

        model = MixtureOfExperts().to(device)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.L1Loss()


        best_val_mae_in_run = float('inf')
        best_model_state_in_run = None
        best_epoch_in_run = -1

        start_time = time.time()

        for epoch in range(NUM_EPOCHS):

            model.train()
            train_mae_sum = 0
            total_train_samples = 0

            for inputs, labels in train_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                final_prediction, _ = model(inputs)


                loss = criterion(final_prediction, labels)
                loss.backward()
                optimizer.step()

                batch_mae_sum = torch.abs(final_prediction - labels).sum().item()
                train_mae_sum += batch_mae_sum
                total_train_samples += labels.size(0)


            model.eval()
            val_errors = []
            with torch.no_grad():
                for inputs, labels in val_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    preds, _ = model(inputs)

                    errors = torch.abs(preds - labels).cpu().numpy()
                    val_errors.extend(errors)

            current_val_mae = np.mean(val_errors)

            if current_val_mae < best_val_mae_in_run:
                best_val_mae_in_run = current_val_mae
                best_epoch_in_run = epoch + 1
                best_model_state_in_run = copy.deepcopy(model.state_dict())

            if (epoch + 1) % 50 == 0:
                avg_train_mae = train_mae_sum / total_train_samples
                print(f"  Epoch {epoch + 1:03d} | Train MAE: {avg_train_mae:.4f} | Val MAE: {current_val_mae:.4f}")

        duration = time.time() - start_time


        if best_model_state_in_run is not None:
            save_path = os.path.join(WEIGHTS_DIR, f'run_{run + 1}_best.pth')
            torch.save(best_model_state_in_run, save_path)

            print(f"  -> Best model obtained at Epoch {best_epoch_in_run}, Val MAE: {best_val_mae_in_run:.4f}")
            print(f"  -> Weights saved: {save_path}")
        else:
            print(f"  Run {run + 1} Exception: Best model not found。")




if __name__ == '__main__':
    train_main()
