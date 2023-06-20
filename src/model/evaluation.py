import numpy as np
import torch
from scipy.stats import ttest_1samp
from scipy.stats.stats import Ttest_1sampResult

from util.statistics import confidence_to_sigma


def compute_mae(model, X_test, y_test, y_scaler):
    """
    Compute the mean absolute error for a given model with test data and a scaler.

    :param model: The nn.Module to evaluate
    :param X_test: Test data features
    :param y_test: Test data labels
    :param y_scaler: Scaler for labels
    :return: MAE
    """
    model.eval()
    with torch.inference_mode():
        pred = model(X_test)
        ae = (pred[0] - y_test).abs() * y_scaler.std
        mae = ae.mean(dim=0)
        return mae


def rejection_rate(ae: np.ndarray, std: np.ndarray, confidence_interval: float = 0.95) -> float:
    """
    Computes the fraction of spots which will be rejected given the absolute errors and their predicted uncertainties.

    A spot is rejected when its absolute error is outside the desired confidence interval.

    :param ae: Numpy array of absolute error scores of the predictions.
    :param std: Numpy array of predicted uncertainties as Gaussian standard deviation of the same shape as ae.
    :param confidence_interval: The confidence interval to use for rejection. Optional, default: 0.95.
    :return: Rejection rate in [0, 1]
    """
    confidence_sigmas = confidence_to_sigma(confidence_interval)
    return len(ae[confidence_sigmas*std < ae]) / len(ae)


def get_rejection_p_samples(ae: np.ndarray, std: np.ndarray, spot_count: int = 4000, samples: int = 1000,
                            confidence_interval: float = 0.95) -> (list, list):
    """
    Samples a given number of spots (spot_count) from the absolute errors (ae) of a number of predictions and their
    uncertainties as Gaussian standard deviation (std) a given number of times (samples) to compute the rejection rates
    and the p-values of a one-sample one-sided t-test with the population mean 0.05.

    :param ae: Numpy array of absolute error scores of a number of predictions.
    :param std: Numpy array of predicted uncertainties as Gaussian standard deviation of the same shape as ae.
    :param spot_count: The number of spots to sample in each iteration.
    :param samples: The number of samples to draw.
    :param confidence_interval: The confidence interval to use for rejection rate and t-test computation.
    :return: Tuple containing a list of rejection rates and a list of p-values for the sampled runs.
    """
    rrs = []
    pvalues = []
    indices = np.arange(ae.shape[0])
    for _ in range(samples):
        sample_indices = np.random.choice(indices, spot_count, replace=False)
        sample_rej = confidence_to_sigma(confidence_interval) * std[sample_indices] < ae[sample_indices]
        sample_rej = np.array(sample_rej).astype(float)
        rrs.append(np.sum(sample_rej) / sample_rej.shape[0])
        test_result: Ttest_1sampResult = ttest_1samp(sample_rej, 0.05, alternative="greater")
        pvalues.append(float(test_result.pvalue))
    return rrs, pvalues
