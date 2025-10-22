import argparse
import json
import os
import pathlib
import pickle
from typing import Optional
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.evaluation import compute_mae, rejection_rate, get_rejection_p_samples
from model.nets import RegressionNet
from model.training import load_or_train_model
from model.uncertainty import get_monte_carlo_predictions, fit_uncertainty_calibrator
from util.config import RunConfig, SingleTaskConfig, MLConfig
from util.determinism import make_deterministic
from util.file_utils import prepare_run_directory
from util.plot_utils import apply_style, save_fig, plot_uncertainties_epi_alea, plot_uncertainties, \
    plot_shifted_rejection_rates, plot_uncertainty_calibration, plot_predictions, plot_error_histogram, \
    plot_errors_in_true_intervals, plot_shifted_pvalues
from util.preprocessing import df_to_dataset, dataset_to_torch
from util.statistics import confidence_to_sigma
from util.torch_utils import TorchStandardScaler


def eval_shifted_spots(task_dir: str, run_config: RunConfig, model, X_scaler, y_scaler, feature_names, targets=None,
                       calibrators=None) -> tuple[dict, dict]:
    if targets is None:
        targets = ["water_range", "range_shifted"]

    df_shifted = pd.read_csv(run_config.test_shifted_data_path)
    remove_columns = [col for col in df_shifted.columns if col not in feature_names and col not in targets]

    distances = ((df_shifted["beam_spot_x"] - df_shifted["beam_spot_x_shifted"]) ** 2
                 + (df_shifted["beam_spot_y"] - df_shifted["beam_spot_y_shifted"]) ** 2) ** 0.5
    X_shifted, y_shifted = df_to_dataset(df_shifted, targets, remove_columns, False)
    X_shifted = X_scaler.transform(torch.tensor(X_shifted, dtype=torch.float32, device=run_config.device))
    y_shifted = y_scaler.transform(torch.tensor(y_shifted, dtype=torch.float32, device=run_config.device))
    shifted_loader = DataLoader(TensorDataset(X_shifted, y_shifted), batch_size=512)

    mean, std, std_epi, std_alea, ae, mae, ground_truth = get_monte_carlo_predictions(model, shifted_loader, y_scaler)

    distance_stats = {}
    for d in distances.unique():
        distance_stats[d] = []
    for i in range(X_shifted.shape[0]):
        distance_stats[distances[i]].append((std_epi[i], std_alea[i], std[i], ae[i]))

    for k, v in distance_stats.items():
        distance_stats[k] = np.array(v)

    results = {}
    fig, axes = plt.subplots(int(np.ceil(len(distance_stats)/2)), 2, figsize=(10, 4*len(distance_stats)/2))
    for k, v in sorted(distance_stats.items(), key=lambda x: x[0]):
        ax = axes[int((k-1) / 2)][int((k-1) % 2)]
        distance_results = {}
        for i in range(v.shape[2]):
            rr_epi = rejection_rate(v[:, 3, i], v[:, 0, i], run_config.rr_ci)
            rr_alea = rejection_rate(v[:, 3, i], v[:, 1, i], run_config.rr_ci)
            rr_combined = rejection_rate(v[:, 3, i], v[:, 2, i], run_config.rr_ci)
            distance_results[f"rr_{i}_epistemic"] = rr_epi
            distance_results[f"rr_{i}_aleatoric"] = rr_alea
            distance_results[f"rr_{i}"] = rr_combined
            if calibrators is not None:
                ci_calib = calibrators[i].predict(np.array([run_config.rr_ci]))
                rr_calib = rejection_rate(v[:, 3, i], v[:, 2, i], ci_calib)
                distance_results[f"rr_{i}_calib"] = rr_calib
        plot_uncertainties(v[:, 3, v.shape[2]-1], v[:, 2, v.shape[2]-1], ax=ax)
        results[int(k)] = distance_results
        ax.set_title(f"{k} mm lateral displacement")
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes[int(np.ceil(len(distance_stats)/2))-1, 1].set_xlabel("Absolute error (mm)")
    axes[0, 0].set_ylabel(f"${confidence_to_sigma(run_config.rr_ci):.2f}\\sigma$")
    save_fig(fig, os.path.join(task_dir, "uncertainties_shifted"))

    return results, distance_stats


def eval_other(run_config: RunConfig, model, X_scaler, y_scaler, feature_names, targets=None,
               calibrators=None):
    results = {}
    if targets is None:
        targets = ["water_range", "range_shifted"]

    targets_regular = [t.replace("_shifted", "") for t in targets]

    df_other = pd.read_csv(run_config.test_other_data_path)
    remove_columns = [col for col in df_other.columns if col not in feature_names and col not in targets_regular]

    X_other, y_other = df_to_dataset(df_other, targets_regular, remove_columns, False)
    X_other = X_scaler.transform(torch.tensor(X_other, dtype=torch.float32, device=run_config.device))
    y_other = y_scaler.transform(torch.tensor(y_other, dtype=torch.float32, device=run_config.device))
    other_loader = DataLoader(TensorDataset(X_other, y_other), batch_size=512)

    mean, std, std_epi, std_alea, ae, mae, ground_truth = get_monte_carlo_predictions(model, other_loader, y_scaler)
    rmse = np.sqrt(np.mean(ae**2, axis=0))
    for i in range(len(targets)):
        results[f"rmse_{i}"] = rmse[i]
        results[f"mae_{i}"] = mae[i].item()
        results[f"rr_{i}"] = rejection_rate(ae[:, i], std[:, i], run_config.rr_ci)
        results[f"rr_{i}_epistemic"] = rejection_rate(ae[:, i], std_epi[:, i], run_config.rr_ci)
        results[f"rr_{i}_aleatoric"] = rejection_rate(ae[:, i], std_alea[:, i], run_config.rr_ci)

        if calibrators is not None:
            ci_calib = calibrators[i].predict(np.array([run_config.rr_ci]))
            rr_calib = rejection_rate(ae[:, i], std[:, i], ci_calib)
            results[f"rr_{i}_calib"] = rr_calib


    # shifted
    df_shifted = pd.read_csv(run_config.test_other_shifted_data_path)
    remove_columns = [col for col in df_shifted.columns if col not in feature_names and col not in targets]

    distances = ((df_shifted["beam_spot_x"] - df_shifted["beam_spot_x_shifted"]) ** 2
                 + (df_shifted["beam_spot_y"] - df_shifted["beam_spot_y_shifted"]) ** 2) ** 0.5
    X_shifted, y_shifted = df_to_dataset(df_shifted, targets, remove_columns, False)
    X_shifted = X_scaler.transform(torch.tensor(X_shifted, dtype=torch.float32, device=run_config.device))
    y_shifted = y_scaler.transform(torch.tensor(y_shifted, dtype=torch.float32, device=run_config.device))
    shifted_loader = DataLoader(TensorDataset(X_shifted, y_shifted), batch_size=512)

    mean, std, std_epi, std_alea, ae, mae, ground_truth = get_monte_carlo_predictions(model, shifted_loader, y_scaler)

    distance_stats = {}
    for d in distances.unique():
        distance_stats[d] = []
    for i in range(X_shifted.shape[0]):
        distance_stats[distances[i]].append((std_epi[i], std_alea[i], std[i], ae[i]))

    for k, v in distance_stats.items():
        distance_stats[k] = np.array(v)

    shifted_results = {}
    for k, v in sorted(distance_stats.items(), key=lambda x: x[0]):
        distance_results = {}
        for i in range(v.shape[2]):
            rr_epi = rejection_rate(v[:, 3, i], v[:, 0, i], run_config.rr_ci)
            rr_alea = rejection_rate(v[:, 3, i], v[:, 1, i], run_config.rr_ci)
            rr_combined = rejection_rate(v[:, 3, i], v[:, 2, i], run_config.rr_ci)
            distance_results[f"rr_{i}_epistemic"] = rr_epi
            distance_results[f"rr_{i}_aleatoric"] = rr_alea
            distance_results[f"rr_{i}"] = rr_combined
            if calibrators is not None:
                ci_calib = calibrators[i].predict(np.array([run_config.rr_ci]))
                rr_calib = rejection_rate(v[:, 3, i], v[:, 2, i], ci_calib)
                distance_results[f"rr_{i}_calib"] = rr_calib
        shifted_results[int(k)] = distance_results

    results["shifted"] = shifted_results
    return results


def eval_task(run_config: RunConfig, task_config: MLConfig,
              X_train, X_val, X_test, y_train, y_val, y_test,
              X_scaler: TorchStandardScaler, y_scaler: TorchStandardScaler,
              feature_names, writer: Optional[SummaryWriter] = None):
    task_dir = os.path.join(run_config.workdir, task_config.model_name)
    pathlib.Path(task_dir).mkdir(exist_ok=True)

    make_deterministic(run_config.seed)

    targets = ["water_range", "range_shifted"]
    task_names = ["R", "Z"]
    if isinstance(task_config, SingleTaskConfig):
        targets = ["water_range"] if task_config.target == "r" else ["range_shifted"]
        task_names = task_config.target.upper()

        target_index = 0 if task_config.target == "r" else 1
        y_scaler_single = TorchStandardScaler(y_train)
        y_train, y_test = y_train[:, target_index:target_index+1], y_test[:, target_index:target_index+1]
        if y_val is not None:
            y_val = y_val[:, target_index:target_index+1]
        y_scaler_single.mean = y_scaler.mean[target_index:target_index+1]
        y_scaler_single.std = y_scaler.std[target_index:target_index+1]
        y_scaler = y_scaler_single

    num_y = len(targets)
    train_loader = DataLoader(TensorDataset(X_train, y_train), task_config.batch_size, shuffle=True, drop_last=True)
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(TensorDataset(X_val, y_val), 512)
    test_loader = DataLoader(TensorDataset(X_test, y_test), 512)

    model = RegressionNet(X_train.shape[1], num_y, task_config.dropout, task_config.activation).to(run_config.device)
    model = load_or_train_model(task_dir, model, run_config, task_config, train_loader, test_loader, y_scaler, writer)

    mae = compute_mae(model, X_test, y_test, y_scaler)
    results = {}
    for i in range(num_y):
        results[f"mae_{i}_plain"] = mae[i].item()

    std_val, std_epi_val, std_alea_val, ae_val = None, None, None, None
    if val_loader is not None:
        _, std_val, std_epi_val, std_alea_val, ae_val, _, _ = get_monte_carlo_predictions(model, val_loader, y_scaler)
    mean, std, std_epi, std_alea, ae, mae, gt = get_monte_carlo_predictions(model, test_loader, y_scaler)
    rmse = np.sqrt(np.mean(ae**2, axis=0))
    for i in range(num_y):
        results[f"rmse_{i}"] = rmse[i]
        results[f"mae_{i}"] = mae[i].item()
        results[f"rr_{i}_epistemic"] = rejection_rate(ae[:, i], std_epi[:, i], run_config.rr_ci)
        results[f"rr_{i}_aleatoric"] = rejection_rate(ae[:, i], std_alea[:, i], run_config.rr_ci)
        results[f"rr_{i}"] = rejection_rate(ae[:, i], std[:, i], run_config.rr_ci)
        if val_loader is not None:
            calibrator = fit_uncertainty_calibrator(ae_val[:, i], std_val[:, i])
            ci_calib = calibrator.predict(np.array([run_config.rr_ci]))
            results[f"rr_{i}_calib"] = rejection_rate(ae[:, i], std[:, i], ci_calib)

    for i in range(num_y):
        save_fig(plot_predictions(mean[:, i], gt[:, i], task_names[i]), os.path.join(task_dir, f"predictions_{i}"))
        save_fig(plot_error_histogram(mean[:, i], gt[:, i]), os.path.join(task_dir, f"error_histogram_{i}"))
        save_fig(plot_errors_in_true_intervals(ae[:, i], gt[:, i]), os.path.join(task_dir, f"mae_in_bins_{i}"))

        calibrators = None
        if val_loader is not None:
            calibrators = [fit_uncertainty_calibrator(ae_val[:, i], std_epi_val[:, i]),
                           fit_uncertainty_calibrator(ae_val[:, i], std_alea_val[:, i])]
        save_fig(
            plot_uncertainty_calibration(
                2 * [ae[:, i]], [std_epi[:, i], std_alea[:, i]], ["Epistemic", "Aleatoric"],
                calibrators
            ),
            os.path.join(task_dir, f"uncertainty_calib_epialea_{i}")
        )

    save_fig(plot_uncertainties_epi_alea(ae, std_epi, std_alea, task_names), os.path.join(task_dir, "uncertainty"))

    calibrators = None
    if val_loader is not None:
        calibrators = [fit_uncertainty_calibrator(ae_val[:, i], std_val[:, i]) for i in range(num_y)]
    save_fig(plot_uncertainty_calibration(ae.T, std.T, task_names, calibrators),
             os.path.join(task_dir, "uncertainty_calib"))

    results_shifted, distance_stats = eval_shifted_spots(task_dir, run_config, model, X_scaler, y_scaler,
                                                         feature_names, targets, calibrators)
    results["shifted"] = results_shifted

    for i in range(num_y):
        rr = [[0, results[f"rr_{i}_epistemic"], results[f"rr_{i}_aleatoric"], results[f"rr_{i}"]]]
        rr += [[k, v[f"rr_{i}_epistemic"], v[f"rr_{i}_aleatoric"], v[f"rr_{i}"]] for k, v in results_shifted.items()]
        rr_data = np.array(rr)
        fig = plot_shifted_rejection_rates(rr_data[:, 0], rr_data[:, 1:], ["Epistemic", "Aleatoric", "Combined"])
        save_fig(fig, os.path.join(task_dir, f"rejection_rates_{i}"))

        if val_loader is not None:
            rr = [[0, results[f"rr_{i}"], results[f"rr_{i}_calib"]]]
            rr += [[k, v[f"rr_{i}"], v[f"rr_{i}_calib"]] for k, v in results_shifted.items()]
            rr_data = np.array(rr)
            fig = plot_shifted_rejection_rates(rr_data[:, 0], rr_data[:, 1:], ["Uncalibrated", "Calibrated"])
            save_fig(fig, os.path.join(task_dir, f"rejection_rates_{i}_calib"))

    distance_stats[0] = np.stack([std_epi, std_alea, std, ae], axis=1)

    spots = np.arange(run_config.ttest.spots_min, run_config.ttest.spots_max+1, run_config.ttest.spots_step)
    ci = run_config.rr_ci
    if val_loader is not None:
        ci = calibrators[-1].predict([ci])[0]
    pvalues = []
    for s in tqdm(spots):
        pvalues_spot = []
        for k, v in sorted(distance_stats.items(), key=lambda x: x[0]):
            _, shift_pvalues = get_rejection_p_samples(v[:, 3, -1], v[:, 2, -1], s, run_config.ttest.samples, ci)
            pvalues_spot.append(shift_pvalues)
        pvalues.append(np.mean(np.array(pvalues_spot), -1))
    pvalues = np.stack(pvalues)

    save_fig(plot_shifted_pvalues(spots, pvalues), os.path.join(task_dir, f"ttest_spot_p"))

    if run_config.test_other_data_path is not None:
        results_other = eval_other(run_config, model, X_scaler, y_scaler, feature_names, targets, calibrators)
        results_other_shifted = results_other["shifted"]
        results["other"] = results_other

        for i in range(num_y):
            rr = [[0, results_other[f"rr_{i}_epistemic"], results_other[f"rr_{i}_aleatoric"], results_other[f"rr_{i}"]]]
            rr += [[k, v[f"rr_{i}_epistemic"], v[f"rr_{i}_aleatoric"], v[f"rr_{i}"]] for k, v in results_other_shifted.items()]
            rr_data = np.array(rr)
            fig = plot_shifted_rejection_rates(rr_data[:, 0], rr_data[:, 1:], ["Epistemic", "Aleatoric", "Combined"])
            save_fig(fig, os.path.join(task_dir, f"rejection_rates_{i}_other"))

            if val_loader is not None:
                rr = [[0, results_other[f"rr_{i}"], results_other[f"rr_{i}_calib"]]]
                rr += [[k, v[f"rr_{i}"], v[f"rr_{i}_calib"]] for k, v in results_other_shifted.items()]
                rr_data = np.array(rr)
                fig = plot_shifted_rejection_rates(rr_data[:, 0], rr_data[:, 1:], ["Uncalibrated", "Calibrated"])
                save_fig(fig, os.path.join(task_dir, f"rejection_rates_{i}_other_calib"))

    with open(os.path.join(task_dir, "results.json"), "w") as results_file:
        json.dump(results, results_file)

    return results


def eval_run_config(run_config: RunConfig):
    prepare_run_directory(run_config)

    df_train = pd.read_csv(run_config.train_data_path)
    df_test = pd.read_csv(run_config.test_data_path)
    X_train, y_train, feature_names = df_to_dataset(df_train)
    X_val, y_val = None, None
    if run_config.val_data_path is not None:
        df_val = pd.read_csv(run_config.val_data_path)
        X_val, y_val = df_to_dataset(df_val, return_feature_names=False)
    X_test, y_test = df_to_dataset(df_test, return_feature_names=False)

    X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler = dataset_to_torch(
        X_train, X_test, y_train, y_test, run_config.device, X_val, y_val
    )

    x_scaler_path = os.path.join(run_config.workdir, "x_scaler.pkl")
    if not os.path.exists(x_scaler_path):
        with open(x_scaler_path, "wb") as f:
            pickle.dump(X_scaler, f)

    y_scaler_path = os.path.join(run_config.workdir, "y_scaler.pkl")
    if not os.path.exists(y_scaler_path):
        with open(y_scaler_path, "wb") as f:
            pickle.dump(y_scaler, f)

    writer = SummaryWriter()
    writer.add_text("config", str(run_config))

    rejection_rate_keys = []

    results = {}
    for task_config in run_config.multitask:
        results[task_config.model_name] = eval_task(
            run_config, task_config,
            X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler, feature_names, writer
        )
        rejection_rate_keys.append(task_config.model_name)

    for task_config in run_config.single_task:
        results[task_config.model_name] = eval_task(
            run_config, task_config,
            X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler, feature_names, writer
        )
        if task_config.target == "z":
            rejection_rate_keys.append(task_config.model_name)

    shift_amount = 10
    rejection_rate_data = np.zeros((shift_amount+1, len(rejection_rate_keys)))
    rejection_rate_data_calib = np.zeros((shift_amount+1, len(rejection_rate_keys)))
    for i, k in enumerate(rejection_rate_keys):
        key = "rr_0"
        if "rr_1" in results[k]:
            key = "rr_1"
        rejection_rate_data[0, i] = results[k][key]
        if run_config.val_data_path is not None:
            rejection_rate_data_calib[0, i] = results[k][key + "_calib"]
        shifted_data = results[k]["shifted"]
        for d in shifted_data.keys():
            rejection_rate_data[d, i] = shifted_data[d][key]
            if run_config.val_data_path is not None:
                rejection_rate_data_calib[d, i] = shifted_data[d][key + "_calib"]
    rejection_rate_labels = []
    for k in rejection_rate_keys:
        components = k.split("_")
        components = [c.upper() if len(c) == 1 else c for c in components]
        label = " ".join(components)
        label = label[:1].upper() + label[1:]
        rejection_rate_labels.append(label)
    fig = plot_shifted_rejection_rates(np.arange(shift_amount+1), rejection_rate_data, rejection_rate_labels)
    save_fig(fig, os.path.join(run_config.workdir, "rejection_rates"))
    fig = plot_shifted_rejection_rates(np.arange(shift_amount+1), rejection_rate_data_calib, rejection_rate_labels)
    save_fig(fig, os.path.join(run_config.workdir, "rejection_rates_calib"))

    with open(os.path.join(run_config.workdir, "results.json"), "w") as results_file:
        json.dump(results, results_file)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate range verification")
    parser.add_argument("-c", dest="config_file", type=str, help="Path to config file to use.")
    parser.add_argument("-w", dest="workdir", type=str, help="Path to directory to save results in.")
    parser.add_argument("-d", dest="device", type=str, required=False, help="Device to use for PyTorch.")
    parser.add_argument("-s", dest="seed", type=int, required=False, help="Random seed to use for evaluation.")
    parser.add_argument("--purge", dest="purge", required=False, action="store_true",
                        help="If specified, workdir is purged before running, overriding value from the config file.")
    args = parser.parse_args()

    apply_style()

    config = RunConfig.from_file(args.config_file)
    config.workdir = args.workdir
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    if args.purge:
        config.purge_workdir = True
    eval_run_config(config)
