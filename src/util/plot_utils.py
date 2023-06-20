import pathlib
from typing import Sequence, List, Optional
import SimpleITK
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from util.statistics import confidence_to_sigma


def apply_style(notebook=False):
    """
    Used to change the matplotlib style to the ATLAS theme from mplhep and adjusts the color cycler to a more
    colorblind-friendly palette from https://gist.github.com/thriveth/8560036.

    Optionally, if this is supposed to be used in Jupyter notebooks in PyCharm, the parameter needs to be set to True
    and then an empty dummy plot will be generated, which for some reason really applies the style.

    :param notebook: If used in Jupyter notebooks, set to True to make sure it applies. Optional, default: False
    """
    plt.style.use(hep.style.ATLAS)

    if notebook:
        plt.figure(figsize=(1,1))
        plt.show()
        plt.style.use(hep.style.ATLAS)

    mpl.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#377eb8", "#ff7f00", "#4daf4a",
                                                           "#f781bf", "#a65628", "#984ea3",
                                                           "#999999", "#e41a1c", "#dede00"])


def save_fig(fig, path: str, close: bool = True):
    """
    Saves a matplotlib figure at a given path. If the path does not include a file extension, two files are saved,
    one as PDF and one as EPS.
    If the figure is needed after saving, the `close` argument must be set to False, so the figure is left open.

    ! Automatic EPS export temporarily disabled !

    :param fig: The figure to save.
    :param path: The path to save the figure to (with filename and optional file extension).
    :param close: Should the figure be closed after saving? Default: True.
    """
    fig.tight_layout()
    # If an extension is already provided, just save at the path
    if pathlib.Path(path).suffix != "":
        fig.savefig(path)
    # Without extension, we save .pdf and .eps
    else:
        fig.savefig(path + ".pdf")
        # fig.savefig(path + ".eps")
    if close:
        plt.close(fig)


def plot_predictions(y_pred: Sequence, y_true: Sequence, target_name: str = "") -> plt.Figure:
    """
    Creates a scatter plot for the predicted values over the ground truth targets.

    :param y_pred: Sequence of predictions.
    :param y_true: Sequence of true values.
    :param target_name: Name of the target variable to put into axis labels.
    :return: The created matplotlib figure.
    """
    fig = plt.figure(figsize=(9, 8))
    plt.scatter(y_true, y_pred, s=1)
    plt.plot([0, 200], [0, 200], color="black", linestyle="dashed", linewidth=1)
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")

    true_min = np.min(y_true)
    true_max = np.max(y_true)
    plt.xlim(true_min - 10, true_max + 10)
    plt.ylim(true_min - 10, true_max + 10)

    fig.tight_layout()
    return fig


def plot_error_histogram(y_pred: np.ndarray, y_true: np.ndarray) -> plt.Figure:
    """
    Creates a histogram plot of the error distribution of the given predictions compared to the given true values.

    :param y_pred: Numpy array of predictions.
    :param y_true: Numpy array of ground truth values.
    :return: The created matplotlib figure.
    """
    fig = plt.figure(figsize=(9, 8))
    plt.hist(y_pred - y_true, bins=np.arange(-15, 15.01, 0.2))
    plt.xlabel("Prediction error (mm)")
    plt.ylabel("Number of predictions in bin")
    plt.xlim(-10, 10)
    fig.tight_layout()
    return fig


def plot_errors_in_true_intervals(ae: np.ndarray, y_true: np.ndarray) -> plt.Figure:
    """
    Splits the true values into intervals of width 1 and computes the mean absolute error in each of the intervals.
    The result is plotted as a scatter plot.

    :param ae: Numpy array of absolute error scores of the predictions.
    :param y_true: Numpy array of true values of the predictions.
    :return: The created matplotlib figure.
    """
    intervals = []
    x = np.arange(int(np.min(y_true)), int(np.max(y_true)) + 1)
    for i in x:
        errors_in_interval = ae[(i <= y_true) & (y_true < i + 1)]
        if len(errors_in_interval) > 0:
            intervals.append((i + 0.5, errors_in_interval.mean()))
    intervals = np.array(intervals)
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(intervals[:, 0], intervals[:, 1], s=10)
    plt.xlabel("Ground truth (mm)")
    plt.ylabel("Mean absolute error in bin (mm)")
    fig.tight_layout()
    return fig


def plot_uncertainties(ae: np.ndarray, std: np.ndarray, confidence_interval: float = 0.95, ax=None,
                       **kwargs) -> plt.Figure:
    """
    Produces a scatter plot, optionally in a given Axes, of the predicted confidence sigmas over the absolute error.

    :param ae: Array of absolute errors.
    :param std: Array of predicted standard deviation.
    :param confidence_interval: Desired confidence interval in range [0, 1] from which the sigma factor is derived.
        Optional, default: 0.95
    :param ax: Optional Axes to plot into.
    :param kwargs: Additional kwargs to pass along to the matplotlib scatter command.
    :return: The resulting figure (either newly created or the one associated with the given Axes).
    """
    factor = confidence_to_sigma(confidence_interval)
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        fig.set_tight_layout(True)
        ax = plt.gca()
    ax.scatter(ae, factor * std, s=1, **kwargs)
    ax.plot([-10, 100], [-10, 100], c="red")
    ax.set_xlim(-0.05, max(np.max(ae), 2 * np.max(std)) + 0.2)
    ax.set_ylim(-0.05, max(np.max(ae), 2 * np.max(std)) + 0.2)
    ax.set_xlabel("Absolute error (mm)")
    ax.set_ylabel(f"{int(round(confidence_interval*100))}% confidence interval")
    return ax.figure


def plot_uncertainties_epi_alea(ae: np.ndarray, std_epi: np.ndarray, std_alea: np.ndarray, target_names: List[str],
                                std_combined: Optional[np.ndarray]=None, confidence_interval=0.95) -> plt.Figure:
    """
    Creates a figure with 3 subplots per given target where the uncertainties are scattered over the absolute error,
    similar to plot_uncertainties, which is used to produce the individual subplots.

    The 3 plots are separated into epistemic, aleatoric, and combined uncertainties.
    Epistemic and aleatoric uncertainties have to be provided, while the combined uncertainty is optional and will be
    automatically computed through sqrt(std_epi**2 + std_alea**2).

    All given uncertainties as well as the absolute errors have to be of the same shape where the first dimension is
    the number of samples N and the second dimension is the number of targets T = len(target_names).

    The confidence interval to use for the uncertainty value can be adjusted as in plot_uncertainties.

    :param ae: Numpy ndarray of absolute errors of shape (N, T).
    :param std_epi: Numpy ndarray of epistemic uncertainties (standard deviation) of shape (N, T).
    :param std_alea: Numpy ndarray of aleatoric uncertainties (standard deviation) of shape (N, T).
    :param target_names: List of names to use in the subplot titles for the individual targets.
    :param std_combined: Numpy ndarray of combined uncertainties (standard deviation) of shape (N, T).
        Optional, default: None, which means computed through sqrt(std_epi**2 + std_alea**2).
    :param confidence_interval: Desired confidence interval in range [0, 1] from which the sigma factor is derived.
        Optional, default: 0.95
    :return: The resulting figure.
    """
    if std_combined is None:
        std_combined = np.sqrt(std_epi**2 + std_alea**2)

    targets = len(target_names)
    fig, axes = plt.subplots(3, targets, figsize=(6*targets, 16))
    axes = axes.flatten()

    for i in range(targets):
        ax = axes[i]
        plot_uncertainties(ae[:, i], std_epi[:, i], confidence_interval=confidence_interval, ax=ax)
        ax.set_xlabel("")
        if i != 0:
            ax.set_ylabel("")
        ax.set_title(f"Epistemic Uncertainty ({target_names[i]})")

        ax = axes[targets + i]
        plot_uncertainties(ae[:, i], std_alea[:, i], confidence_interval=confidence_interval, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Aleatoric Uncertainty ({target_names[i]})")

        ax = axes[2*targets + i]
        plot_uncertainties(ae[:, i], std_combined[:, i], confidence_interval=confidence_interval, ax=ax)
        if i != targets-1:
            ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Combined Uncertainty ({target_names[i]})")

    fig.tight_layout()
    return fig


def plot_uncertainty_calibration(ae: Sequence[np.ndarray], std: Sequence[np.ndarray], labels: List[str],
                                 calibrators: Optional[list] = None) -> plt.Figure:
    """
    Generates a curve for a given list of absolute errors and their predicted uncertainties in the form of
    Gaussian standard deviation. The curve computes the observed confidence, i.e., how many predictions fall into the
    interval, for 100 expected confidence levels (1%, 2%, ...).

    The plot is then a line between 0,0 and 1,1 with the observed over the expected confidence interval.

    A list of labels is used to assign a label to the second dimension of the input data. The list length n must match
    the second dimension of ae and std.

    Optional calibrators can be given to draw additional curves after the calibrators are applied. There need to be
    n calibrators, if given. The calibrators are expected to transform an expected confidence level into a
    confidence level to use on the data instead, in order to match the expectation. This works through the
    calibrator's predict method, which maps [0, 1] -> [0, 1].

    :param ae: Absolute error scores of shape (x, n).
    :param std: Standard deviation of the Gaussian uncertainty of shape (x, n).
    :param labels: List of n labels to use in the plots for the datasets.
    :param calibrators: Optional n calibrators to use for confidence interval adjustment of the datasets.
    :return: The created matplotlib figure.
    """
    if len(ae) != len(std) or len(ae) != len(labels) or calibrators is not None and len(ae) != len(calibrators):
        raise ValueError("All given sequences must be of the same length")

    # Get the sigma factors for all confidence intervals to plot through the PPF of a normal distribution
    # with mu=0 and sigma=1.
    x = np.linspace(0, 1, 100)[:-1]

    fig = plt.figure(figsize=(9, 8))

    for i in range(len(ae)):
        sigma_factors = confidence_to_sigma(x)
        ar = [len(ae[i][ae[i] <= sigma_factor * std[i]]) / len(ae[i]) for sigma_factor in sigma_factors] + [1]
        plt.plot(np.append(x, 1), ar, label=labels[i])
        if calibrators is not None:
            sigma_factors = confidence_to_sigma(calibrators[i].predict(x))
            ar = [len(ae[i][ae[i] <= sigma_factor * std[i]]) / len(ae[i]) for sigma_factor in sigma_factors] + [1]
            plt.plot(np.append(x, 1), ar, label=f"{labels[i]} calibrated")

    plt.plot([0, 1], [0, 1], label="Optimal calibration", c="black", linewidth=1, linestyle="dashed")
    plt.xlabel("Expected confidence level")
    plt.ylabel("Observed confidence level")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.tight_layout()
    return fig


def plot_shifted_rejection_rates(shifts: Sequence, rr_data: np.ndarray, labels: List[str], figsize=(9, 8),
                                 error_bars: np.ndarray = None) -> plt.Figure:
    """
    Creates a scatter plot where the given rejection rates are displayed over their respective shifts in mm.
    Shows three different scatter plots, for epistemic, aleatoric, and combined uncertainty.

    :param shifts: Sequence of values to use for the magnitude of the shift in the data.
    :param rr_data: Numpy ndarray of shape (len(shifts), len(labels)) containing the rejection rates for all
        shift/label combinations.
    :param labels: List of labels to use for the individual rejection rate variants.
    :param figsize: Tuple of width and height in inches to use as size of the figure.
    :param error_bars: Optional numpy ndarray of the same shape as rr_data representing error bar length in
        both directions in y.
    :return: The created matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)

    marker_rotation = ["D", "o", "P", "^"]
    for i in range(rr_data.shape[1]):
        marker = marker_rotation[i % len(marker_rotation)]
        if error_bars is None:
            plt.scatter(shifts, rr_data[:, i], marker=marker, label=labels[i])
        else:
            plt.errorbar(shifts, rr_data[:, i], yerr=error_bars[:, i], marker=marker, linestyle="", label=labels[i])

    plt.xlim(-0.5, len(shifts) - 0.5)
    plt.ylim(0, np.min([np.max(rr_data) + 0.1, 1]))
    plt.xlabel("Lateral shift from planned spot (mm)")
    plt.ylabel("Spot rejection rate")
    plt.grid(color="gray", alpha=0.3)
    plt.grid(which="minor", color="lightgray", alpha=0.3)
    plt.legend(loc="upper left")
    return fig


def plot_shifted_pvalues(spots: Sequence, shifted_pvalues: np.ndarray) -> plt.Figure:
    """
    Plots a list of average p-values from t-tests over different spot counts. shifted_pvalues is assumed to have at
    least size 4 in the second dimension and the indices correspond to the number of mm the samples were shifted.
    Only shifts of 1, 2, and 3 mm are plotted, 0 and everything above 3 are ignored.
    The length of spots needs to match the first dimension in shifted_pvalues.

    :param spots: Sequence of spot values for the x-axis.
    :param shifted_pvalues: Numpy array containing the average p-values with dimension (len(spots), >=4).
    :return: The created matplotlib figure.
    """
    fig = plt.figure(figsize=(9, 8))
    plt.plot(spots, shifted_pvalues[:, 1], label="1 mm shift", linestyle="", marker="D")
    plt.plot(spots, shifted_pvalues[:, 2], label="2 mm shift", linestyle="", marker="o")
    plt.plot(spots, shifted_pvalues[:, 3], label="3 mm shift", linestyle="", marker="P")
    plt.hlines(0.05, 0, 4000, colors="orange", linestyles="dashed", label="$\\alpha=0.05$")
    plt.hlines(0.01, 0, 4000, colors="red", linestyles="dotted", label="$\\alpha=0.01$")
    plt.ylim(0, int(np.ceil(np.max(shifted_pvalues[:, 1]) * 10)) / 10.)
    plt.xlabel("Spots")
    plt.ylabel("p-value")
    plt.legend()
    fig.tight_layout()
    return fig
