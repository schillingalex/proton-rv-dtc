from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from physics.chip import get_cluster_size, get_edep
from util.torch_utils import TorchStandardScaler


def bin_hits(df: pd.DataFrame, group_columns: List[str] = None) -> pd.DataFrame:
    """
    Bins the hits in a given DataFrame by the given group columns, usually eventID, trackID, layer.

    The rows in each bin are then combined by taking the sum of the edep column and the mean of the position columns
    posX, posY, posZ.

    If group_columns is None, by default eventID, trackID, and all available volume IDs are used.

    :param df: Input DataFrame to bin.
    :param group_columns: Columns to group by, usually eventID, trackID, layer.
    :return: Binned DataFrame.
    """
    if group_columns is None:
        group_columns = ["eventID", "trackID"]
        for col in df.columns:
            if col.startswith("volume"):
                group_columns.append(col)

    agg = {}
    for col in df.columns:
        agg[col] = "first"
    agg["posX"] = "mean"
    agg["posY"] = "mean"
    agg["posZ"] = "mean"
    agg["edep"] = "sum"
    return df.groupby(group_columns).aggregate(agg).reset_index(drop=True)


def transform_edep_to_cluster_size(df: pd.DataFrame,
                                   edep_column: str = "edep",
                                   cs_column: str = "clusterSize") -> pd.DataFrame:
    """
    Calls the transformation function turning an energy deposition into a cluster size and adds the corresponding
    column to the given df (inplace). The column names to consider can be adjusted with `edep_column` and `cs_column`.
    The resulting DataFrame is returned.

    :param df: The DataFrame to do the transformation in. Has to contain a column named like the param `edep_column`.
    :param edep_column: The column to use as input energy deposition. Optional, default: "edep".
    :param cs_column: The column to add, containing the cluster sizes. Optional, default: "clusterSize".
    :return: The resulting DataFrame.
    """
    df[cs_column] = get_cluster_size(df[edep_column].values)
    return df


def transform_cluster_size_to_edep(df: pd.DataFrame,
                                   edep_column: str = "edep",
                                   cs_column: str = "clusterSize") -> pd.DataFrame:
    """
    Calls the transformation function turning a cluster size into an energy deposition estimate and adds the
    corresponding column to the given df (inplace). The column names to consider can be adjusted with `edep_column` and
    `cs_column`.

    :param df: The DataFrame to do the transformation in. Has to contain a column named like the param `cs_column`.
    :param edep_column: The column to add, containing the energy deposition values. Optional, default: "edep".
    :param cs_column: The column to use as input cluster sizes. Optional, default: "clusterSize".
    :return: The resulting DataFrame.
    """
    df[edep_column] = get_edep(df[cs_column].values)
    return df


def df_to_dataset(df: pd.DataFrame,
                  target_columns: Optional[List[str]] = None,
                  remove_columns: Optional[List[str]] = None,
                  return_feature_names=True) \
        -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Turns a DataFrame containing features, as well as ground truth information, into two numpy.ndarrays X, y,
    where X contains the features and y the labels.

    The feature columns are filtered by their standard deviation: features with std < prune_std_threshold are removed.

    :param df: Source DataFrame containing all required columns for features and target values.
    :param target_columns: List of column names to use as target values. Optional, default: ["water_range", "range"]
    :param remove_columns: List of column names to remove from the features, but which are not the target value.
        Optional, default: ["x", "y", "range_straggling", "lateral_deflection", "energy"]
    :param return_feature_names: Whether to return a list with the resulting feature names. Optional, default: True
    :return: The features, the target values, and optionally the feature names
    """
    if remove_columns is None:
        remove_columns = ["x", "y", "range_straggling", "lateral_deflection", "energy",
                          "beam_spot_x", "beam_spot_y", "phantom_rotation_angle"]
    if target_columns is None:
        target_columns = ["water_range", "range"]

    remove_columns += target_columns

    feature_names = [c for c in df.columns if c not in remove_columns]
    X = df[feature_names].values
    y = df[target_columns].values

    if return_feature_names:
        return X, y, feature_names
    else:
        return X, y


def dataset_to_torch(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, device=None,
                     X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> tuple:
    """
    Turns a dataset consisting of features and labels for train and test sets into torch tensors on the specified
    device. Additionally, the data is normalized to mean=0, std=1 and mean and std from the training set are returned.

    :param X_train: The training features as numpy.ndarray.
    :param X_test: The test features as numpy.ndarray.
    :param y_train: The training labels as numpy.ndarray.
    :param y_test: The test labels as numpy.ndarray.
    :param device: The device to put the tensor on. Optional, default: None.
    :param X_val: Optional validation features as numpy.ndarray.
    :param y_val: Optional validation labels as numpy.ndarray.
    :return: Tuple of torch.Tensor containing the following:
        Train features, test features, train labels, test labels,
        TorchStandardScaler for X and y
    """
    import torch

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    X_scaler = TorchStandardScaler(X_train)
    X_train, X_test = X_scaler.transform(X_train), X_scaler.transform(X_test)

    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    y_scaler = TorchStandardScaler(y_train, keepdim=False)
    y_train, y_test = y_scaler.transform(y_train), y_scaler.transform(y_test)

    if X_val is not None and y_val is not None:
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
        X_val, y_val = X_scaler.transform(X_val), y_scaler.transform(y_val)
        return X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler
