from scipy.stats import norm


def confidence_to_sigma(confidence_interval):
    """
    Computes the factor to use for sigma to get the desired confidence interval in a unit Gaussian.

    Example: 0.95 confidence -> 1.96 sigma.

    :param confidence_interval: Scalar or numpy ndarray of confidence intervals to transform into sigma factors.
    :return: The sigma factors for the given confidence interval(s).
    """
    return norm.ppf(0.5 + confidence_interval / 2)
