import os
import json
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from cluster.clustering import add_cluster_labels, generate_cluster_df
from cluster.diffusion import Diffuser
from physics.rsp_image import MetaImageRSPImage
from util.curve_fits import fits_over_distribution, fit_gaussian

pd.options.mode.chained_assignment = None


def extract_ground_truth_from_stopping(filename: str):
    """
    Implementation of Pettersen 2018 method to find range distribution data based on a
    ProductionAndStoppingActor in GATE, more specifically the stopping information.

    The stopping voxel image is assumed to have voxel size 1 mm and volume 200x200x200 mm^3.

    :param filename: Name of the MetaImage file containing the stopping count voxels or a metadata file containing
        an "output" dictionary with the "phantom_stop_files" key. Only one file is expected in "phantom_stop_files".
        If more are given, the first one is used.
    :return: Ground truth information extracted from the given stopping information.
    """
    rotation_angle = 0
    # Figure out the file from the metadata
    if filename.endswith(".json"):
        with open(filename) as meta_file:
            metadata = json.load(meta_file)
            if "output" not in metadata.keys():
                raise ValueError("Meta file does not contain output.")
            if "phantom_stop_files" not in metadata["output"].keys():
                raise ValueError("Meta file does not contain ProductionAndStoppingActor output")
            if len(metadata["output"]["phantom_stop_files"]) == 0:
                raise ValueError("Meta file does not contain any stopping information files")
            filename = os.path.join(os.path.dirname(filename), metadata["output"]["phantom_stop_files"][0])

            if "parameters" not in metadata.keys():
                raise ValueError("Meta file does not contain parameters.")
            if "phantom_rotation_angle" in metadata["parameters"].keys():
                rotation_angle = float(metadata["parameters"]["phantom_rotation_angle"])

    voxel_size = 1.
    volume = np.array([200, 200, 200])

    voxel_image = MetaImageRSPImage(
        filename, transform_hu_to_rsp=False,
        volume=volume, voxel_size=voxel_size, rotation_angle=rotation_angle,
        align_to_y=False, flip_x_y=True, padding_value=0
    )
    world_voxels = voxel_image.get_world_voxels()

    # Range (z-coordinate) and straggling
    stops_z = []
    for i in range(world_voxels.shape[2]):
        stops_z.append(np.sum(world_voxels[:, :, i]))
    x = np.arange(len(stops_z)) + voxel_size/2
    y = np.array(stops_z)
    popt_z, _, _ = fit_gaussian(x, y)
    # Range = Mean of fitted Gaussian
    r = popt_z[1]
    # Range straggling = Sigma of fitted Gaussian
    sigma_r = popt_z[2]

    # Lateral deflection / transverse beam spread and x-coordinate
    stops_x = []
    for i in range(world_voxels.shape[0]):
        stops_x.append(np.sum(world_voxels[i, :, :]))
    x = np.arange(len(stops_x)) + voxel_size/2
    y = np.array(stops_x)
    popt_x, _, _ = fit_gaussian(x, y)
    pos_x = popt_x[1] - volume[0]/2
    sigma_x_over_r = popt_x[2] / r

    # y-coordinate
    stops_y = []
    for i in range(world_voxels.shape[1]):
        stops_y.append(np.sum(world_voxels[:, i, :]))
    x = np.arange(len(stops_y)) + voxel_size/2
    y = np.array(stops_y)
    popt_y, _, _ = fit_gaussian(x, y)
    pos_y = popt_y[1] - volume[1]/2

    return {
        "x": pos_x,
        "y": pos_y,
        "range": r,
        "range_straggling": sigma_r,
        "lateral_deflection": sigma_x_over_r
    }


def extract_features_pseudopixels(df: pd.DataFrame, base_features: Optional[dict] = None) -> dict:
    """
    Extracts features from a dataset which has not been transformed into pixels, but instead has a transformation of
    the energy deposition values into cluster sizes given in column "clusterSize".

    Additionally, the given data has to be preprocessed in order to contain a "layer" column, specifying the DTC
    layer from 0 through 42.

    :param df: The preprocessed data to extract features from.
    :param base_features: Optional dict with base features to add the generated features to.
    :return: A dictionary containing the base features, if any, as well as the newly generated features.
    """
    layer_column = "layer"

    features = {}
    if isinstance(base_features, dict):
        features.update(base_features)

    features["pixels"] = df["clusterSize"].sum()
    features["clusters"] = len(df)

    features["clusters_over_5"] = len(df[df["clusterSize"] >= 5])
    features["clusters_over_20"] = len(df[df["clusterSize"] >= 20])

    features["cluster_size_mean"] = df["clusterSize"].mean()
    features["cluster_size_std"] = df["clusterSize"].std()

    df["clusterSize"] = np.clip(df["clusterSize"], None, 72)
    cs_hist, _ = np.histogram(df["clusterSize"].values, bins=np.arange(1, 74) - 0.5)
    for i in range(len(cs_hist)):
        features[f"cs_{i+1}"] = cs_hist[i]

    features["x_mean"] = df["posX"].mean()
    features["x_std"] = df["posX"].std()
    features["y_mean"] = df["posY"].mean()
    features["y_std"] = df["posY"].std()

    layers = list(range(43))

    pixels_dist = [sum(df[df[layer_column] == i]["clusterSize"]) for i in layers]
    features.update(fits_over_distribution(layers, pixels_dist, "pixels", normalize=False))

    clusters_dist = [len(df[df[layer_column] == i]) for i in layers]
    features.update(fits_over_distribution(layers, clusters_dist, "clusters", normalize=False))

    edep_dist = [sum(df[df[layer_column] == i]["edep"]) for i in layers]
    features.update(fits_over_distribution(layers, edep_dist, "edep", normalize=False))

    for i in range(len(pixels_dist)):
        features[f"pixels_layer_{i}"] = pixels_dist[i]
    for i in range(len(clusters_dist)):
        features[f"clusters_layer_{i}"] = clusters_dist[i]
    for i in range(len(edep_dist)):
        features[f"edep_layer_{i}"] = edep_dist[i]

    layer_groups = df.groupby(layer_column)
    x_mean_dist = layer_groups["posX"].mean()
    x_std_dist = layer_groups["posX"].std()
    y_mean_dist = layer_groups["posY"].mean()
    y_std_dist = layer_groups["posY"].std()

    for i in range(len(x_mean_dist)):
        features[f"x_mean_layer_{i}"] = x_mean_dist[i]
    for i in range(len(x_std_dist)):
        features[f"x_std_layer_{i}"] = x_std_dist[i]
    for i in range(len(y_mean_dist)):
        features[f"y_mean_layer_{i}"] = y_mean_dist[i]
    for i in range(len(y_std_dist)):
        features[f"y_std_layer_{i}"] = y_std_dist[i]

    return features


def extract_features_pixels(df: pd.DataFrame, diffuser: Diffuser, base_features: Optional[dict] = None) -> dict:
    layer_column = "layer"

    features = {}
    if isinstance(base_features, dict):
        features.update(base_features)

    diffused = diffuser.diffuse_hits(df)
    diffused["group_column"] = diffused["eventID"] * 100 + diffused["layer"]
    clustered = add_cluster_labels(diffused)
    df = generate_cluster_df(clustered)
    df = df[df["size"] > 1]

    features["pixels"] = df["size"].sum()
    features["clusters"] = len(df)

    features["cluster_size_mean"] = df["size"].mean()
    features["cluster_size_std"] = df["size"].std()

    cs_hist, _ = np.histogram(df["size"].values, bins=np.arange(1, 73) + 0.5)
    for i in range(len(cs_hist)):
        features[f"cs_{i+2}"] = cs_hist[i]

    features["x_mean"] = df["posX"].mean()
    features["x_std"] = df["posX"].std()
    features["y_mean"] = df["posY"].mean()
    features["y_std"] = df["posY"].std()

    layers = list(range(43))

    pixels_dist = [sum(df[df[layer_column] == i]["size"]) for i in layers]
    features.update(fits_over_distribution(layers, pixels_dist, "pixels", normalize=False))

    clusters_dist = [len(df[df[layer_column] == i]) for i in layers]
    features.update(fits_over_distribution(layers, clusters_dist, "clusters", normalize=False))

    edep_dist = [sum(df[df[layer_column] == i]["edep"]) for i in layers]
    features.update(fits_over_distribution(layers, edep_dist, "edep", normalize=False))

    for i in range(len(pixels_dist)):
        features[f"pixels_layer_{i}"] = pixels_dist[i]
    for i in range(len(clusters_dist)):
        features[f"clusters_layer_{i}"] = clusters_dist[i]
    for i in range(len(edep_dist)):
        features[f"edep_layer_{i}"] = edep_dist[i]

    layer_groups = df.groupby(layer_column)
    x_mean_dist = layer_groups["posX"].mean()
    x_std_dist = layer_groups["posX"].std()
    y_mean_dist = layer_groups["posY"].mean()
    y_std_dist = layer_groups["posY"].std()

    for i in range(len(x_mean_dist)):
        features[f"x_mean_layer_{i}"] = x_mean_dist[i]
    for i in range(len(x_std_dist)):
        features[f"x_std_layer_{i}"] = x_std_dist[i]
    for i in range(len(y_mean_dist)):
        features[f"y_mean_layer_{i}"] = y_mean_dist[i]
    for i in range(len(y_std_dist)):
        features[f"y_std_layer_{i}"] = y_std_dist[i]

    return features


def extract_features_rsp(rsp: np.ndarray,
                         beam_spot_x: float, beam_spot_sigma_x: float,
                         beam_spot_y: float, beam_spot_sigma_y: float) -> dict:
    """
    Extracts features from a given 3D-RSP image based on a beam spot and size.

    The beam is assumed to be parallel to the z-axis.
    The rsp image is assumed to be rotated correctly to have the z-axis point along the beam axis.

    The extracted features are a number of weighted RSP values for each individual slice (i.e. size of axis 2 in rsp)
    and an additional total RSP, the sum of all others.

    :param rsp: 3D-RSP image with 1 mm voxel size.
    :param beam_spot_x: x-coordinate of the pencil beam (mm).
    :param beam_spot_sigma_x: Standard deviation of the gaussian pencil beam in x-direction (mm).
    :param beam_spot_y: y-coordinate of the pencil beam (mm).
    :param beam_spot_sigma_y: Standard deviation of the gaussian pencil beam in y-direction (mm).
    :return: Dictionary containing the RSP slices and total RSP value extracted from the RSP image.
    """
    features = {}

    beam_gauss_gen = multivariate_normal([beam_spot_x, beam_spot_y],
                                         [[beam_spot_sigma_x ** 2, 0], [0, beam_spot_sigma_y ** 2]])
    beam_gauss = lambda x, y: beam_gauss_gen.pdf(np.dstack((x, y)))

    res_x = rsp.shape[0] // 2
    res_y = rsp.shape[1] // 2
    res_z = int(rsp.shape[2])
    x, y = np.meshgrid(np.arange(-res_x, res_x, 1), np.arange(-res_y, res_y, 1))
    rsps = np.sum(rsp[:, :, :] * beam_gauss(x.T + 0.5, y.T + 0.5)[:, :, None], axis=(0, 1))
    for z in range(res_z):
        features[f"RSP_{z}"] = rsps[z]
    features["RSP_total"] = rsps.sum()

    return features
