import os
import numpy as np
import pandas as pd
import cv2
import random
import matplotlib.pyplot as plt
import hashlib


BASE_SEED = 42

DATASET_ROOT = r'....\my_dataset_split\data'

TRAIN_FOLDER = os.path.join(DATASET_ROOT, 'train')
VAL_FOLDER = os.path.join(DATASET_ROOT, 'val')
TEST_FOLDER = os.path.join(DATASET_ROOT, 'test')


CSV_PATH = r'.......\detection_data.csv'

OUTPUT_FOLDER = 'case34'
PARAMS_FILENAME = 'final_params.txt'
RESULTS_FILENAME = 'case34.csv'
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'fit_plots')

for folder in [OUTPUT_FOLDER, PLOTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)


SMALL_CIRCLE_RADIUS_OPTIONS = [10, 15, 20]
DEGREE_OPTIONS = [2, 3, 4, 5, 6]
SAMPLING_ZONE_RATIO = 0.5
MIN_CIRCLES = 5
MAX_CIRCLES = 7


LAMBDA1_NM, LAMBDA2_NM = 565, 632
C2 = 1.43877e-2
lambda1_m, lambda2_m = LAMBDA1_NM * 1e-9, LAMBDA2_NM * 1e-9


def stable_seed(*items):
    s = '|'.join(map(str, items)).encode('utf-8')
    return (BASE_SEED ^ int(hashlib.sha256(s).hexdigest()[:8], 16)) & 0xffffffff


def calculate_theoretical_radiance_ratio(T, lambda1_m, lambda2_m):
    if T <= 0: return 0
    return (lambda2_m / lambda1_m) ** 5 * np.exp((C2 / T) * (1 / lambda2_m - 1 / lambda1_m))


def invert_temperature(measured_radiance_ratio, lambda1_m, lambda2_m):
    if measured_radiance_ratio <= 0: return 0
    ratio_term = (lambda2_m / lambda1_m) ** 5
    if ratio_term <= 0 or measured_radiance_ratio / ratio_term <= 0: return 0
    ln_term = np.log(measured_radiance_ratio / ratio_term)
    denominator = (1 / lambda2_m - 1 / lambda1_m)
    if denominator == 0 or ln_term == 0: return 0
    return C2 * denominator / ln_term


def calculate_aligned_roi_averages(image, r_tl, r_tr, small_circle_radius, sampling_zone_ratio, min_circles,
                                   max_circles, rng, is_training=True):
    w1, h1 = r_tl['bbox_x2'] - r_tl['bbox_x1'], r_tl['bbox_y2'] - r_tl['bbox_y1']
    cx1, cy1 = r_tl['bbox_x1'] + w1 / 2, r_tl['bbox_y1'] + h1 / 2
    w2, h2 = r_tr['bbox_x2'] - r_tr['bbox_x1'], r_tr['bbox_y2'] - r_tr['bbox_y1']
    cx2, cy2 = r_tr['bbox_x1'] + w2 / 2, r_tr['bbox_y1'] + h2 / 2

    min_w, min_h = min(w1, w2), min(h1, h2)


    if is_training:
        max_sampling_radius = (min(min_w, min_h) / 2.0) * sampling_zone_ratio
        num_circles = 1 if max_sampling_radius <= small_circle_radius else rng.randint(min_circles, max_circles)

        relative_offsets = [(0, 0)]

        for _ in range(num_circles - 1):
            for _ in range(50):
                r = np.sqrt(rng.random()) * (max_sampling_radius - small_circle_radius)
                theta = rng.random() * 2 * np.pi
                offset_x, offset_y = r * np.cos(theta), r * np.sin(theta)
                is_overlapping = False
                for ex, ey in relative_offsets:
                    if np.sqrt((offset_x - ex) ** 2 + (offset_y - ey) ** 2) < 2 * small_circle_radius:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    relative_offsets.append((offset_x, offset_y))
                    break
        current_radius = small_circle_radius

    else:
        test_radius_ratio = 0.2
        calculated_radius = int((min(min_w, min_h) / 2.0) * test_radius_ratio)
        current_radius = max(1, calculated_radius)
        relative_offsets = [(0, 0)]


    averages1, averages2 = [], []
    for offset_x, offset_y in relative_offsets:

        c1x, c1y = int(cx1 + offset_x), int(cy1 + offset_y)
        mask1 = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask1, (c1x, c1y), current_radius, 255, -1)
        roi_pixels1 = image[mask1 == 255]

        if roi_pixels1.size > 0:
            valid_pixels1 = roi_pixels1[(roi_pixels1 > 10) & (roi_pixels1 < 250)]
            val = np.mean(valid_pixels1) if valid_pixels1.size > 0 else np.mean(roi_pixels1)
            averages1.append(val)


        c2x, c2y = int(cx2 + offset_x), int(cy2 + offset_y)
        mask2 = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask2, (c2x, c2y), current_radius, 255, -1)
        roi_pixels2 = image[mask2 == 255]
        if roi_pixels2.size > 0:
            valid_pixels2 = roi_pixels2[(roi_pixels2 > 10) & (roi_pixels2 < 250)]
            val = np.mean(valid_pixels2) if valid_pixels2.size > 0 else np.mean(roi_pixels2)
            averages2.append(val)

    avg1 = np.mean(averages1) if averages1 else 0
    avg2 = np.mean(averages2) if averages2 else 0
    return avg1, avg2


def process_dataset(image_folder, df_metadata, small_radius, dataset_type='train'):
    is_training = (dataset_type == 'train')

    calibration_data = []

    base_file_groups = (
        df_metadata[['temperature', 'base_filename']]
        .drop_duplicates()
        .sort_values('base_filename')
    )

    for _, group_info in base_file_groups.iterrows():
        temp_K, base_name = group_info['temperature'], group_info['base_filename']
        image_group_df = df_metadata[df_metadata['base_filename'] == base_name]

        gray1_vals, gray2_vals = [], []


        for image_name in sorted(image_group_df['filename'].unique()):
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path): continue

            try:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None: continue
                image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

                single_img_df = image_group_df[image_group_df['filename'] == image_name]
                row_tl = single_img_df[single_img_df['position'] == 'bottom_left']
                row_tr = single_img_df[single_img_df['position'] == 'bottom_right']

                if row_tl.empty or row_tr.empty: continue
                rng = random.Random(stable_seed(base_name, image_name, small_radius))

                avg_gray1, avg_gray2 = calculate_aligned_roi_averages(
                    image_blurred, row_tl.iloc[0], row_tr.iloc[0], small_radius,
                    SAMPLING_ZONE_RATIO, MIN_CIRCLES, MAX_CIRCLES, rng=rng,
                    is_training=is_training
                )

                if avg_gray1 > 0 and avg_gray2 > 0:
                    gray1_vals.append(avg_gray1)
                    gray2_vals.append(avg_gray2)

            except Exception:
                pass

        if not gray1_vals or not gray2_vals: continue


        avg_g1, avg_g2 = np.mean(gray1_vals), np.mean(gray2_vals)
        if avg_g2 == 0: continue

        calibration_data.append({
            'base_filename': base_name,
            'true_temp_K': temp_K,
            'x_theoretical': calculate_theoretical_radiance_ratio(temp_K, lambda1_m, lambda2_m),
            'y_measured': avg_g1 / avg_g2
        })

    return pd.DataFrame(calibration_data)


def plot_calibration_curve(train_df, val_df, coeffs, title, output_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    font_size_title, font_size_label, font_size_legend = 24, 20, 18

    ax.scatter(train_df['y_measured'], train_df['x_theoretical'],
               color='green', alpha=0.6, label='Train Data', s=60)


    ax.scatter(val_df['y_measured'], val_df['x_theoretical'],
               color='blue', marker='x', label='Val Data', s=100)


    all_y = pd.concat([train_df['y_measured'], val_df['y_measured']])
    if not all_y.empty:
        y_curve = np.linspace(all_y.min(), all_y.max(), 200)
        x_curve = np.polyval(coeffs, y_curve)
        ax.plot(y_curve, x_curve, color='red', linewidth=3, label=f'Polynomial Fit (Degree {len(coeffs) - 1})')

    ax.set_xlabel('Grayscale Ratio', fontsize=font_size_label)
    ax.set_ylabel('Radiance Ratio (Theoretical)', fontsize=font_size_label)
    ax.tick_params(axis='both', which='major', labelsize=font_size_label-4)

    ax.legend(fontsize=font_size_legend)
    ax.grid(True)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"error: {e}")
    plt.close(fig)


if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        exit(f"error'{CSV_PATH}'")

    df_source = pd.read_csv(CSV_PATH)

    try:
        train_filenames = set(os.listdir(TRAIN_FOLDER))
        val_filenames = set(os.listdir(VAL_FOLDER))
        test_filenames = set(os.listdir(TEST_FOLDER))
    except FileNotFoundError as e:
        exit(f"error {e}")


    df_train_meta = df_source[df_source['filename'].isin(train_filenames)].copy()
    df_val_meta = df_source[df_source['filename'].isin(val_filenames)].copy()
    df_test_meta = df_source[df_source['filename'].isin(test_filenames)].copy()


    for df in [df_train_meta, df_val_meta, df_test_meta]:
        df['base_filename'] = df['filename'].apply(lambda x: x.rsplit('-', 1)[0])


    best_params = {}
    lowest_mae_on_val = float('inf')


    for small_radius in SMALL_CIRCLE_RADIUS_OPTIONS:
        print(f"\n[R={small_radius}]")


        train_df = process_dataset(TRAIN_FOLDER, df_train_meta, small_radius, dataset_type='train')

        val_df = process_dataset(VAL_FOLDER, df_val_meta, small_radius, dataset_type='val_test')

        if train_df.empty or val_df.empty:
            print(" error ")
            continue


        for degree in DEGREE_OPTIONS:
            if len(train_df) < degree + 1: continue


            coeffs = np.polyfit(train_df['y_measured'], train_df['x_theoretical'], degree)


            val_errors = []
            for _, row in val_df.iterrows():
                x_calc = np.polyval(coeffs, row['y_measured'])
                temp_K = invert_temperature(x_calc, lambda1_m, lambda2_m)
                if temp_K > 0:
                    val_errors.append(abs(temp_K - row['true_temp_K']))

            if not val_errors: continue

            current_mae = np.mean(val_errors)
            current_rmse = np.sqrt(np.mean(np.square(val_errors)))



            if current_mae < lowest_mae_on_val:
                lowest_mae_on_val = current_mae
                best_params = {
                    'degree': degree,
                    'radius': small_radius,
                    'coeffs': coeffs,
                    'train_df': train_df.copy(),
                    'val_df': val_df.copy()
                }


    if not best_params: exit("No valid model found.")

    print(f"Optimal parameters -> radius: {best_params['radius']}, degree: {best_params['degree']}")
    print(f"Corresponding Val MAE: {lowest_mae_on_val:.4f} K")

    final_coeffs = best_params['coeffs']
    final_train_df = best_params['train_df']
    final_val_df = best_params['val_df']


    plot_calibration_curve(
        final_train_df, final_val_df, final_coeffs,
        f"",
        os.path.join(PLOTS_FOLDER, 'best_model_curve.png')
    )

    test_df = process_dataset(TEST_FOLDER, df_test_meta, best_params['radius'], dataset_type='val_test')

    all_results = []

    datasets_to_eval = [
        ('Train', final_train_df),
        ('Val', final_val_df),
        ('Test', test_df)
    ]

    summary_metrics = {}

    for name, df in datasets_to_eval:
        if df.empty: continue

        errors = []
        for _, row in df.iterrows():
            x_calc = np.polyval(final_coeffs, row['y_measured'])
            temp_K, err_K, rel_err = np.nan, np.nan, np.nan

            if x_calc > 0:
                temp_K = invert_temperature(x_calc, lambda1_m, lambda2_m)
                if temp_K > 0:
                    err_K = temp_K - row['true_temp_K']
                    rel_err = (err_K / row['true_temp_K']) * 100
                    errors.append(err_K)


            all_results.append({
                'dataset_type': name,
                'base_filename': row['base_filename'],
                'original_temperature_K': row['true_temp_K'],
                'theoretical_radiance_ratio': row['x_theoretical'],
                'measured_gray_ratio_12': row['y_measured'],
                'calculated_temperature_K': temp_K,
                'absolute_error_K': err_K,
                'relative_error_%': rel_err
            })


        if errors:
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(np.square(errors)))
            summary_metrics[name] = {'MAE': mae, 'RMSE': rmse}
            print(f"  {name} -> MAE: {mae:.4f} K | RMSE: {rmse:.4f} K")


    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_FOLDER, RESULTS_FILENAME), index=False, float_format='%.4f')

    with open(os.path.join(OUTPUT_FOLDER, PARAMS_FILENAME), 'w', encoding='utf-8') as f:
        f.write("--- Final Model Parameters ---\n")
        f.write(f"Radius (for Train sampling): {best_params['radius']}\n")
        f.write(f"Degree: {best_params['degree']}\n")
        f.write("\n--- Sampling Strategy ---\n")
        f.write("Train: Random Multi-Circle\n")
        f.write("Val & Test: Fixed Center 0.2 Region\n")
        f.write("\n--- Model Coefficients (High to Low) ---\n")
        f.write(", ".join([str(c) for c in final_coeffs]) + "\n")
        f.write("\n--- Performance Summary ---\n")
        for name, metrics in summary_metrics.items():
            f.write(f"{name}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}\n")

    print(f"\nAll completed! Results saved to: {OUTPUT_FOLDER}")
