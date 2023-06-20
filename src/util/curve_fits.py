import numpy as np
from functools import partial
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from typing import Tuple, Callable, Iterable


def distribution_rmse(x: list, y: list, distribution: Callable[[float], float]):
    """
    Computes root mean squared error for a given function evaluated at the points in x compared to the values in y.

    x and y need to be of equal lengths.

    :param x: Comparison points.
    :param y: Comparison values.
    :param distribution: Function to evaluate given x points.
    :return: Root mean squared error of given function evaluated at points x compared with y.
    """
    squared_errors = [(distribution(xi) - yi)**2 for xi, yi in zip(x, y)]
    return np.sqrt(np.mean(squared_errors))


def fits_over_distribution(x: list, y: list, base_name: str, normalize: bool = True) -> dict:
    """
    Creates a linear, cubic, and exponential fit over the given input data `x` and `y`.

    The resulting coefficients are returned in a dictionary where the keys are prepended by `base_name`
    and are named by the type of fit "lin", "cubic", "exp" with the coefficient index or "rmse" for the root mean
    squared error of the fit.

    The exponential fit is done through a linear fit for log(y).

    :param x: x-values for the data to fit.
    :param y: y-values for the data to fit.
    :param base_name: Prefix for the resulting dictionary keys.
    :param normalize: Normalize the data by dividing the individual y-values by their sum?
    :return: A dictionary containing the coefficients of the function fits over the data.
    """
    # Replace NaNs with 0.
    y = [0 if np.isnan(yi) else yi for yi in y]

    fits = {}

    # Normalizing the histogram
    if normalize:
        y = list(np.array(y) / np.sum(y))

    lin_coef = Polynomial.fit(x, y, 1).convert().coef
    fits[base_name + "_lin_0"] = lin_coef[0]
    fits[base_name + "_lin_1"] = lin_coef[1]
    fits[base_name + "_lin_rmse"] = distribution_rmse(x, y, Polynomial(lin_coef))

    cube_coef = Polynomial.fit(x, y, 3).convert().coef
    fits[base_name + "_cubic_0"] = cube_coef[0]
    fits[base_name + "_cubic_1"] = cube_coef[1]
    fits[base_name + "_cubic_2"] = cube_coef[2]
    fits[base_name + "_cubic_3"] = cube_coef[3]
    fits[base_name + "_cubic_rmse"] = distribution_rmse(x, y, Polynomial(cube_coef))

    popt_exp, exp_func = fit_exponential(x, y)
    fits[base_name + "_exp_a"] = popt_exp[0]
    fits[base_name + "_exp_b"] = popt_exp[1]
    fits[base_name + "_exp_rmse"] = distribution_rmse(x, y, exp_func)
    return fits


def fit_gaussian(x: Iterable, y: Iterable) -> Tuple[np.ndarray, np.ndarray, Callable[[float], float]]:
    """
    Performs a Gaussian fit over the given data.

    Adopted from: https://stackoverflow.com/a/38431524

    :param x: The x values to use for the fit
    :param y: The y values to use for the fit, corresponding to the given x values
    :return: A tuple with the optimal parameters for the fit, the covariance, and the fitted function.
    """
    x = np.array(x)
    y = np.array(y)

    def gauss(xi, a, mu, sigma):
        return a * np.exp(-(xi - mu)**2 / (2 * sigma**2))

    mu_initial = sum(x * y) / sum(y)
    sigma_initial = np.sqrt(sum(y * (x - mu_initial)**2) / sum(y))

    popt, pcov = curve_fit(gauss, x, y, p0=[np.max(y), mu_initial, sigma_initial], maxfev=10000)
    return popt, pcov, partial(gauss, a=popt[0], mu=popt[1], sigma=popt[2])


def fit_exponential(x: Iterable, y: Iterable) -> Tuple[np.ndarray, Callable[[float], float]]:
    """
    Fits an exponential function over the given data x,y by performing a linear fit in log space.

    :param x: The x values to use for the fit.
    :param y: The y values to use for the fit, corresponding to the given x values.
    :return: A tuple with the fit parameters and the fitted function as a callable.
    """
    def exp_func(xi, a, b):
        return np.exp(a + b*xi)

    # Exp fit is computationally too unstable. We do a linear fit in log space instead.
    # See https://stackoverflow.com/a/3433503/19299651
    popt = Polynomial.fit(x, np.log(y), 1, w=np.sqrt(y)).convert().coef
    return popt, partial(exp_func, a=popt[0], b=popt[1])
