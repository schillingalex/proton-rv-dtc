import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.statistics import confidence_to_sigma
from util.torch_utils import TorchStandardScaler, enable_dropout


def get_monte_carlo_predictions(model: nn.Module, data_loader: DataLoader, forward_passes: int,
                                y_scaler: TorchStandardScaler):
    """
    Takes a PyTorch model with dropout and evaluates the data coming from the given data loader by turning on
    dropout in inference mode and sampling the prediction forward_passes times in order to estimate model uncertainty
    of the prediction. The given model is expected to predict the value and the variance representing data uncertainty.

    :param model: The PyTorch model to evaluate.
    :param data_loader: The data loader containing the data to evaluate.
    :param forward_passes: The number of stochastic forward passes to sample.
    :param y_scaler: A fitted scaler used to transform the predicted value back to human-interpretable results.
    :return: mean, std, std_epi, std_alea, ae, mae, ground_truth
        mean: Means of the sampled predictions of each example in the dataset.
        std: Standard deviation representing total uncertainty (data + model).
        std_epi: Standard deviation representing epistemic (model) uncertainty.
        std_area: Standard deviation representing aleatoric (data) uncertainty.
        ae: Absolute errors of the mean predictions.
        mae: Mean absolute error score over the entire dataset.
        ground_truth: Ground truth labels from the data loader.
    """
    if data_loader.drop_last:
        n_samples = len(data_loader) * data_loader.batch_size
    else:
        n_samples = len(data_loader.dataset)

    ground_truths = []
    dropout_preds = np.empty((forward_passes, n_samples, y_scaler.mean.shape[0]))
    dropout_uncerts = np.empty((forward_passes, n_samples, y_scaler.mean.shape[0]))
    model.eval()
    enable_dropout(model)
    with torch.inference_mode():
        for i, (X, y) in tqdm(list(enumerate(data_loader))):
            ground_truths.append((y_scaler.inverse_transform(y)).cpu().numpy())
            preds = []
            uncerts = []
            for _ in range(forward_passes):
                output = model(X)
                output_pred = y_scaler.inverse_transform(output[0])
                output_uncert = torch.sqrt(output[1]) * y_scaler.std
                preds.append(output_pred.cpu().numpy())
                uncerts.append(output_uncert.cpu().numpy())

            dropout_preds[:, i * data_loader.batch_size:i * data_loader.batch_size + X.shape[0], :] = np.stack(preds)
            dropout_uncerts[:, i*data_loader.batch_size:i*data_loader.batch_size + X.shape[0], :] = np.stack(uncerts)

    ground_truth = np.concatenate(ground_truths)
    # Predictions (mean over forward passes)
    mean = np.mean(dropout_preds, axis=0)
    # Standard deviation for each prediction
    std_epi = np.std(dropout_preds, axis=0)
    # Aleatoric uncertainty as predicted by the network
    std_alea = np.mean(dropout_uncerts, axis=0)
    # Total var = epistemic var + aleatoric var
    std = np.sqrt(std_epi ** 2 + std_alea ** 2)
    # Absolute error of each prediction
    ae = np.abs(mean - ground_truth)
    # Mean absolute error over all predictions
    mae = np.mean(ae, axis=0)
    return mean, std, std_epi, std_alea, ae, mae, ground_truth


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
