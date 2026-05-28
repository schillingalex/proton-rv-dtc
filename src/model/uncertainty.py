import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.statistics import confidence_to_sigma
from util.torch_utils import TorchStandardScaler, enable_dropout


def get_monte_carlo_predictions(model: nn.Module, data_loader: DataLoader,
                                y_scaler: TorchStandardScaler, show_progress=True):
    """
    Takes a PyTorch model with dropout and evaluates the data coming from the given data loader by turning on
    dropout in inference mode and sampling the prediction forward_passes times in order to estimate model uncertainty
    of the prediction. The given model is expected to predict the value and the variance representing data uncertainty.

    :param model: The PyTorch model to evaluate.
    :param data_loader: The data loader containing the data to evaluate.
    :param y_scaler: A fitted scaler used to transform the predicted value back to human-interpretable results.
    :param show_progress: Should the progress be shown through a tqdm progress bar? Default: True
    :return: mean, std, std_epi, std_alea, ae, mae, ground_truth
        mean: Means of the sampled predictions of each example in the dataset.
        std: Standard deviation representing total uncertainty (data + model).
        std_epi: Standard deviation representing epistemic (model) uncertainty.
        std_area: Standard deviation representing aleatoric (data) uncertainty.
        ae: Absolute errors of the mean predictions.
        mae: Mean absolute error score over the entire dataset.
        ground_truth: Ground truth labels from the data loader.
    """
    forward_passes = 100

    ground_truths = []
    preds = []
    stds_epi = []
    stds_alea = []

    model.eval()
    enable_dropout(model)
    with torch.inference_mode():
        iterator = data_loader
        if show_progress:
            iterator = tqdm(iterator)
        for X, y in iterator:
            ground_truths.append(y_scaler.inverse_transform(y))

            X_repeat = torch.repeat_interleave(X, repeats=forward_passes, dim=0)
            pred, var_alea = model(X_repeat)

            pred = y_scaler.inverse_transform(pred)
            pred = pred.view(-1, forward_passes, y.shape[-1])
            # Predictions (mean over forward passes)
            mean = pred.mean(dim=1)
            # Epistemic uncertainty = Standard deviation for each prediction
            std_epi = pred.std(dim=1)

            var_alea = var_alea * y_scaler.std**2
            var_alea = var_alea.view(-1, forward_passes, y.shape[-1])
            var_alea = var_alea.mean(dim=1)
            std_alea = torch.sqrt(var_alea)

            preds.append(mean)
            stds_epi.append(std_epi)
            stds_alea.append(std_alea)

    ground_truth = torch.cat(ground_truths).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()
    std_epi = torch.cat(stds_epi).cpu().numpy()
    std_alea = torch.cat(stds_alea).cpu().numpy()

    # Total var = epistemic var + aleatoric var
    std = np.sqrt(std_epi ** 2 + std_alea ** 2)
    # Absolute error of each prediction
    ae = np.abs(preds - ground_truth)
    # Mean absolute error over all predictions
    mae = np.mean(ae, axis=0)
    return preds, std, std_epi, std_alea, ae, mae, ground_truth


def fit_uncertainty_calibrator(ae, std):
    """
    Fit a regressor to calibrate uncertainties according to
    Kuleshov et al. 2018 "Accurate Uncertainties for Deep Learning Using Calibrated Regression".

    The resulting regressor can be used to feed the desired confidence interval to (e.g. 95%) and it predicts the
    confidence interval to use instead (e.g. 92%), to get the correct confidence interval for the dataset at hand.

    :param ae: Absolute errors on the calibration set.
    :param std: Predicted standard deviations on the calibration set.
    :return: Estimator with the capability to predict the confidence interval to use given a desired confidence.
    """
    x = np.linspace(0, 1, 100)[:-1]
    sigma_factors = confidence_to_sigma(x)
    ar = [len(ae[ae < sigma_factor * std]) / len(ae) for sigma_factor in sigma_factors]

    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(ar + [1], np.append(x, 1))
    return calibrator
