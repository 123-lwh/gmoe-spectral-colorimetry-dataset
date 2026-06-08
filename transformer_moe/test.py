import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from model import MixtureOfExperts  


NUM_RUNS = 10
BATCH_SIZE = 32

CSV_FILE_PATH = r'...moe_test_1.csv'
BASE_OUTPUT_DIR = r'...result'

WEIGHTS_DIR = os.path.join(BASE_OUTPUT_DIR, 'weights')


BEST_RUN_TEST_RESULTS_CSV = os.path.join(BASE_OUTPUT_DIR, 'results.csv')
ALL_RUNS_SUMMARY_CSV = os.path.join(BASE_OUTPUT_DIR, 'summary.csv')


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        exit(f"error '{CSV_FILE_PATH}'")

    split_column = df.columns[0]
    df_test = df[df[split_column] == 'Test'].copy()

    if df_test.empty:
        exit("error 'Test' ")


    filenames_test = df_test.iloc[:, 1].values
    real_temperatures_test = df_test.iloc[:, 2].values
    feature_columns = df.columns[3:]
    X_test = torch.tensor(df_test[feature_columns].values, dtype=torch.float32)
    y_test = torch.tensor(real_temperatures_test, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    all_runs_results = []
    best_mae = float('inf')
    best_run_details_df = None



    for run in range(NUM_RUNS):
        weight_path = os.path.join(WEIGHTS_DIR, f'run_{run + 1}_best.pth')

        if not os.path.exists(weight_path):

            continue


        model = MixtureOfExperts().to(device)

        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        all_predictions = []
        all_weights = []

        with torch.no_grad():
            for inputs, _ in test_loader:

                inputs = inputs.to(device)

                predicted_temps, predicted_weights = model(inputs)

                all_predictions.extend(predicted_temps.cpu().numpy().flatten().tolist())
                all_weights.extend(predicted_weights.cpu().numpy().tolist())

        current_results_df = pd.DataFrame({
            'filename': filenames_test,
            'real_temperature': real_temperatures_test,
            'predicted_temperature': all_predictions
        })
        current_results_df['error'] = current_results_df['predicted_temperature'] - current_results_df[
            'real_temperature']

        current_run_rmse = np.sqrt(np.mean(current_results_df['error'] ** 2))
        current_run_mae = np.mean(np.abs(current_results_df['error']))




        all_runs_results.append({
            'run_number': run + 1,
            'mae_on_test_set': current_run_mae,
            'rmse_on_test_set': current_run_rmse,
            'weight_file': weight_path
        })


        if current_run_mae < best_mae:
            best_mae = current_run_mae


            weights_np = np.array(all_weights)
            for i in range(weights_np.shape[1]):
                current_results_df[f'weight_{i + 1}'] = weights_np[:, i]

            current_results_df['relative_error_%'] = np.where(
                current_results_df['real_temperature'] != 0,
                (current_results_df['error'] / current_results_df['real_temperature']) * 100, 0)

            best_run_details_df = current_results_df.copy()

    if all_runs_results:
        summary_df = pd.DataFrame(all_runs_results)
        summary_df.to_csv(ALL_RUNS_SUMMARY_CSV, index=False, float_format='%.4f')




    if best_run_details_df is not None:
        best_run_details_df.to_csv(BEST_RUN_TEST_RESULTS_CSV, index=False, float_format='%.4f')



if __name__ == '__main__':
    main()
