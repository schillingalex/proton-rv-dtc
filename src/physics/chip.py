from typing import Union
import numpy as np


def get_cluster_size(edep: np.ndarray):
    """
    Transforms an array of energy deposition values in MeV into cluster sizes on the ALPIDE chip.
    Power fit from Pettersen 2019 and the implementation in the DTCToolkit repository, transformed from keV/um to MeV.

    :param edep: Array of energy deposition values to convert.
    :return: Array of cluster sizes corresponding to the input energy depositions.
    """
    cs = 4.2267 * (edep*1000/25)**0.65 + 0.5
    return cs.astype(int)


def get_edep(cs: Union[np.ndarray, int]):
    """
    Transforms an array or single integer of cluster sizes on the ALPIDE chip into energy deposition estimates.
    Power fit from Pettersen 2019 and the implementation in the DTCToolkit repository, transformed from keV/um to MeV:
    0.10890 * 25 / 1000 = 0.0027225

    :param cs: Array of cluster sizes or single cluster size integer to convert.
    :return: Array of energy deposition values corresponding to the input cluster sizes.
    """
    return 0.0027225 * cs**1.5384
