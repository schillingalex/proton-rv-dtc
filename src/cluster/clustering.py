from typing import List
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def add_cluster_labels(df: pd.DataFrame,
                       cluster_columns: List[str] = None,
                       cluster_column_name: str = "cluster_id",
                       group_column: str = "group_column",
                       **kwargs) -> pd.DataFrame:
    """
    The given data gets an additional <cluster_column_name> column containing the IDs of the found clusters.
    The ID is unique over the entire dataset.

    group_column can be an empty string if no grouping is required. The groups can be e.g. single detector
    chips, which have to be separated when considering clusters, since they may use a local coordinate system,
    not to be mixed with other chips' local coordinates.

    **kwargs can be used to pass additional arguments to the fit_predict function of the clusterer.

    :param df: Data to cluster. Must contain columns corresponding to what is passed in with the other parameters.
    :param cluster_columns: List of columns to use for clustering. Optional, default: ["column", "row"]
    :param cluster_column_name: Name of the column used for the ID of the assigned cluster.
    :param group_column: Column to group by to cluster separately. Optional, default "frameID".
        Can be an empty string if grouping is not desired.
    :return: The input DataFrame with an additional clusterID column containing cluster IDs.
    """
    if cluster_columns is None:
        cluster_columns = ["column", "row"]

    clusterer = DBSCAN(1, min_samples=1)

    def cluster(group):
        group[cluster_column_name] = clusterer.fit_predict(group[cluster_columns], **kwargs)
        return group

    df[cluster_column_name] = np.arange(len(df)) + 1
    # If we are grouping into separate clustering regions, we need to group_by/apply and then adjust the cluster IDs
    # in order to not have any overlap between different regions.
    if len(group_column) > 0:
        df = df.groupby(group_column, group_keys=False, sort=False).apply(cluster)

        max_cluster_id = df[cluster_column_name].max()
        if max_cluster_id <= 0 or np.isnan(max_cluster_id):
            exponent = 1
        else:
            exponent = int(np.ceil(np.log10(max_cluster_id)))
        offset_factor = 10 ** exponent
        df[cluster_column_name] = df[group_column] * offset_factor + df[cluster_column_name]
    else:
        df = cluster(df)

    df.drop_duplicates(inplace=True)
    return df


def get_size(pixel_identifiers):
    return len(set(pixel_identifiers))


def get_column_centroid(pixel_identifiers):
    a = np.array(list(set(pixel_identifiers)))
    return np.mean(np.floor_divide(a, 1000) % 10000)


def get_row_centroid(pixel_identifiers):
    a = np.array(list(set(pixel_identifiers)))
    return np.mean(a % 1000)


def generate_cluster_df(df: pd.DataFrame,
                        cluster_column_name: str = "cluster_id",
                        pixel_x_column: str = "column",
                        pixel_y_column: str = "row",
                        consider_duplicates: bool = True) -> pd.DataFrame:
    pixel_x_centroid_column = pixel_x_column + "_centroid"
    pixel_y_centroid_column = pixel_y_column + "_centroid"

    agg = {k: pd.NamedAgg(column=k, aggfunc="first") for k in df.columns}

    agg["size"] = pd.NamedAgg(column=cluster_column_name, aggfunc="count")
    agg[pixel_x_centroid_column] = pd.NamedAgg(column=pixel_x_column, aggfunc="mean")
    agg[pixel_y_centroid_column] = pd.NamedAgg(column=pixel_y_column, aggfunc="mean")

    if "PDGEncoding" in df.columns:
        # Largest particle participating in creating the cluster
        agg["PDGEncoding"] = pd.NamedAgg(column="PDGEncoding", aggfunc="max")

    if consider_duplicates:
        df["unique_pixel_identifier"] = (df[pixel_x_column]) * 1000 + df[pixel_y_column]

        agg["size"] = pd.NamedAgg(column="unique_pixel_identifier", aggfunc=get_size)
        agg[pixel_x_centroid_column] = pd.NamedAgg(column="unique_pixel_identifier", aggfunc=get_column_centroid)
        agg[pixel_y_centroid_column] = pd.NamedAgg(column="unique_pixel_identifier", aggfunc=get_row_centroid)

    cluster_df = df.groupby(cluster_column_name, group_keys=False, sort=False).agg(**agg)
    return cluster_df
