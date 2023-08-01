# Limits threads used by numpy and scipy for certain features.
# Must be done before importing the libraries.
# Only one of the two is required, depending on the system,
# but just using both to not worry about where this is executed.
# Source: https://stackoverflow.com/a/48665619/19299651
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import pickle
import time
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.evaluation import rejection_rate
import model.features as f
from model.nets import RegressionNet
from model.uncertainty import get_monte_carlo_predictions
from physics.rsp_image import MetaImageRSPImage
from util.config import RunConfig, MLConfig, SingleTaskConfig
from util.determinism import make_deterministic
from util.plot_utils import apply_style
from util.preprocessing import df_to_dataset
from util.torch_utils import TorchStandardScaler


def eval_feature_generation(data_files):
    rsp_file = "../data/imageDump.mhd"
    rsp_arrays = {}
    rsp_load_times = []
    for pra in tqdm(range(0, 360, 30), "Load RSP"):
        start_time = time.time()
        rsp_image = MetaImageRSPImage(rsp_file, rotation_angle=pra)
        rsp_arrays[pra] = rsp_image.get_world_voxels()
        rsp_load_times.append(time.time() - start_time)
    results = {
        "rsp_load_mean": np.mean(rsp_load_times),
        "rsp_load_std": np.std(rsp_load_times),
    }

    feature_dicts = []

    times_detector_features = []
    times_phantom_features = []
    for metafile_path in tqdm(data_files, "Features"):
        metadata: dict
        with open(metafile_path) as metafile:
            metadata = json.load(metafile)

        base_features = metadata["ground_truth"]

        params = metadata["parameters"]
        beam_spot_x = params["beam_spot_x"]
        beam_spot_sigma_x = params["beam_spot_sigma_x"]
        beam_spot_y = params["beam_spot_y"]
        beam_spot_sigma_y = params["beam_spot_sigma_y"]

        base_features["beam_spot_x"] = beam_spot_x
        base_features["beam_spot_y"] = beam_spot_y

        energy = params["beam_energy"]
        base_features["energy"] = energy
        ranges = pd.read_csv("../data/proton_energies.csv")
        ranges["diff"] = np.abs(ranges["energy"] - energy)
        base_features["water_range"] = ranges["range"][ranges["diff"].argmin()]

        pra = params["phantom_rotation_angle"]
        base_features["phantom_rotation_angle"] = pra

        data_file = os.path.join(os.path.dirname(metafile_path), metadata["output"]["output_files"][0])

        start_time = time.time()
        df = pd.DataFrame(np.load(data_file))
        features = f.extract_features_pseudopixels(df, base_features)
        times_detector_features.append(time.time() - start_time)

        start_time = time.time()
        rsp = rsp_arrays[pra]
        rsp_features = f.extract_features_rsp(rsp, beam_spot_x, beam_spot_sigma_x, beam_spot_y, beam_spot_sigma_y)
        times_phantom_features.append(time.time() - start_time)
        features.update(rsp_features)
        feature_dicts.append(features)

    results.update({
        "detector_features_mean": np.mean(times_detector_features),
        "detector_features_std": np.std(times_detector_features),
        "phantom_features_mean": np.mean(times_phantom_features),
        "phantom_features_std": np.std(times_phantom_features),
    })
    return results, pd.DataFrame(feature_dicts)


def eval_task(workdir, run_config: RunConfig, task_config: MLConfig, X, y, y_scaler: TorchStandardScaler):
    task_dir = os.path.join(workdir, task_config.model_name)
    make_deterministic(run_config.seed)

    num_y = 2
    if isinstance(task_config, SingleTaskConfig):
        num_y = 1

        target_index = 0 if task_config.target == "r" else 1
        y_scaler_single = TorchStandardScaler(y)
        y = y[:, target_index:target_index+1]
        y_scaler_single.mean = y_scaler.mean[target_index:target_index+1]
        y_scaler_single.std = y_scaler.std[target_index:target_index+1]
        y_scaler = y_scaler_single

    model = RegressionNet(X.shape[1], num_y, task_config.dropout, task_config.activation).to(run_config.device)
    model_path = os.path.join(task_dir, f"model.pt")
    model.load_state_dict(torch.load(model_path, map_location=run_config.device))

    results = {}

    mcdo_sample_times = []
    for i in tqdm(range(X.shape[0]), "MCDO"):
        start_time = time.time()
        test_loader = DataLoader(TensorDataset(X[i].unsqueeze(0), y[i].unsqueeze(0)), 1)
        get_monte_carlo_predictions(model, test_loader, run_config.uncertainty.forward_passes, y_scaler, False)
        mcdo_sample_times.append(time.time() - start_time)
    results["mcdo_samples_mean"] = np.mean(mcdo_sample_times)
    results["mcdo_samples_std"] = np.std(mcdo_sample_times)

    test_loader = DataLoader(TensorDataset(X, y), 512)
    mean, std, std_epi, std_alea, ae, mae, gt = get_monte_carlo_predictions(
        model, test_loader, run_config.uncertainty.forward_passes, y_scaler
    )
    rejection_rate_times = []
    for _ in tqdm(range(10000), "rr"):
        start_time = time.time()
        rejection_rate(ae[:, -1], std[:, -1], 0.95)
        rejection_rate_times.append(time.time() - start_time)
    results["rejection_rate_mean"] = np.mean(rejection_rate_times)
    results["rejection_rate_std"] = np.std(rejection_rate_times)

    return results


def eval_run_config(workdir, run_config: RunConfig, results, data):
    x_scaler_path = os.path.join(workdir, "x_scaler.pkl")
    with open(x_scaler_path, "rb") as x_scaler_file:
        X_scaler = pickle.load(x_scaler_file)

    y_scaler_path = os.path.join(workdir, "y_scaler.pkl")
    with open(y_scaler_path, "rb") as y_scaler_file:
        y_scaler = pickle.load(y_scaler_file)

    X, y = df_to_dataset(data, return_feature_names=False)
    X = X_scaler.transform(torch.tensor(X, dtype=torch.float32, device=run_config.device))
    y = y_scaler.transform(torch.tensor(y, dtype=torch.float32, device=run_config.device))

    for task_config in run_config.multitask:
        print(f"Evaluating task {task_config.model_name}")
        results[task_config.model_name] = eval_task(workdir, run_config, task_config, X, y, y_scaler)

    for task_config in run_config.single_task:
        print(f"Evaluating task {task_config.model_name}")
        results[task_config.model_name] = eval_task(workdir, run_config, task_config, X, y, y_scaler)

    with open(os.path.join(workdir, "timing_results.json"), "w") as results_file:
        json.dump(results, results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure timing of the workflow from sim output to range prediction.")
    parser.add_argument("-c", dest="config_file", type=str, help="Path to config file to use.")
    parser.add_argument("-w", dest="workdir", type=str,
                        help="Path to directory where a trained model for the config can be found.")
    parser.add_argument("-d", dest="device", type=str, required=False, help="Device to use for PyTorch.")
    parser.add_argument("-s", dest="seed", type=int, required=False, help="Random seed to use for evaluation.")
    parser.add_argument("data_files", nargs="+", help="The files or file patterns to use for timing evaluations.")
    args = parser.parse_args()

    apply_style()

    print("Globbing...")
    data_files = []
    for data_file_pattern in args.data_files:
        data_files.extend(list(glob(data_file_pattern)))
    print("Files collected")

    results_feature_gen, data = eval_feature_generation(data_files)

    config = RunConfig.from_file(args.config_file)
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    eval_run_config(args.workdir, config, results_feature_gen, data)
