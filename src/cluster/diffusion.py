import os
from abc import ABC, abstractmethod
from functools import partial
from typing import Union
import math
import scipy.stats as st
import numpy as np
import pandas as pd
import torch

from cluster.clustering import add_cluster_labels, generate_cluster_df


def generate_pixel_grid(size=11, xshift: torch.Tensor | float = 0., yshift: torch.Tensor | float = 0.,
                        px_w=0.02924, px_h=0.02688, device=None, centers=False):
    if not isinstance(xshift, torch.Tensor):
        xshift = torch.tensor(xshift, dtype=torch.float32, device=device).unsqueeze(0)
    if not isinstance(yshift, torch.Tensor):
        yshift = torch.tensor(yshift, dtype=torch.float32, device=device).unsqueeze(0)

    x = (torch.arange(0, size+1, device=device) - size / 2.0)[None, :] - xshift[:, None]
    x *= px_w
    y = (torch.arange(0, size+1, device=device) - size / 2.0)[None, :] - yshift[:, None]
    y *= px_h
    xi, yi = torch.meshgrid(torch.arange(size+1, device=device), torch.arange(size+1, device=device), indexing="ij")
    x = x[:, xi]
    y = y[:, yi]
    if centers:
        x = (x[:, 1:, 1:] + x[:, :-1, :-1]) / 2
        y = (y[:, 1:, 1:] + y[:, :-1, :-1]) / 2
    return x, y


def kernel_from_distribution(distribution, eta: Union[float, torch.Tensor] = 1., size=11,
                             xshift: torch.Tensor | float = 0., yshift: torch.Tensor | float = 0.,
                             px_w=0.02924, px_h=0.02688, device=None) -> torch.Tensor:
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, dtype=torch.float32, device=device)
    device = eta.device

    x, y = generate_pixel_grid(size, xshift, yshift, px_w, px_h, device)
    kern2d = ((distribution(x[:, 1:, 1:], y[:, :-1, :-1]) - distribution(x[:, :-1, :-1], y[:, :-1, :-1]))
              - (distribution(x[:, 1:, 1:], y[:, 1:, 1:]) - distribution(x[:, :-1, :-1], y[:, 1:, 1:])))
    kern2d = kern2d/kern2d.sum(dim=(1, 2), keepdim=True) * eta
    return torch.transpose(kern2d, 1, 2)


def integ_bivar_cauchy_func(x, y, gamma):
    # Bivariate Cauchy: 1/(2*torch.pi) * gamma / (x**2 + y**2 + gamma**2)**(3/2)
    # Indefinite integral according to Wolfram Alpha.
    return torch.atan((x*y) / (gamma*torch.sqrt(x**2 + y**2 + gamma**2))) / (2*torch.pi)


def cauchy_gen(gamma=1.):
    return partial(integ_bivar_cauchy_func, gamma=gamma)


class Diffuser(ABC):
    def __init__(self, threshold=92, random_collection=False, device=None, name=None, label=None, batch_size=10_000):
        self.threshold = threshold
        self.random_collection = random_collection
        self.name = name
        self.label = label
        self.batch_size = batch_size

        self.px_w = 0.02924
        self.px_h = 0.02688
        self.px_dims = torch.tensor([self.px_w, self.px_h], device=device)

        self.device = device

    def _pos_to_px(self, df):
        df["column"] = df["posX"] / self.px_w
        df["row"] = df["posY"] / self.px_h
        df["columnShift"] = df["column"] % 1 - 0.5
        df["rowShift"] = df["row"] % 1 - 0.5
        return df

    def _preprocess_hits(self, df) -> pd.DataFrame:
        df["electrons"] = 1e6 / 3.6 * df["edep"]
        df = self._pos_to_px(df)
        return df

    @abstractmethod
    def _turn_hits_into_charges(self, df) -> pd.DataFrame:
        pass

    def diffuse_hits(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._preprocess_hits(df)
        diffused = self._turn_hits_into_charges(df)
        del diffused["electrons"]
        diffused["column"] = diffused["column"].astype(int)
        diffused["row"] = diffused["row"].astype(int)

        if self.random_collection:
            diffused["charge"] = st.poisson.rvs(diffused["charge"])

        agg = {k: "first" for k in diffused.columns}
        agg["charge"] = "sum"
        diffused = diffused.groupby(["eventID", "layer", "column", "row"], as_index=False, sort=False).agg(agg)
        diffused = diffused[diffused["charge"] >= self.threshold]

        del diffused["charge"]
        del diffused["columnShift"]
        del diffused["rowShift"]
        diffused.reset_index(drop=True, inplace=True)
        return diffused

    def __call__(self, df):
        return self.diffuse_hits(df)

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.name


class KernelDiffuser(Diffuser):
    def __init__(self, kernel_func, eta, **kwargs):
        super().__init__(**kwargs)
        self.kernel_func = kernel_func
        self.eta = torch.tensor(eta, device=self.device)
        self.size = 13
        self.half_size = int(self.size/2)

    def _turn_hits_into_charges(self, df) -> pd.DataFrame:
        colshift = torch.from_numpy(df["columnShift"].values).to(self.device)
        rowshift = torch.from_numpy(df["rowShift"].values).to(self.device)
        kernel = kernel_from_distribution(self.kernel_func, self.eta, self.size, colshift, rowshift)
        electrons = torch.from_numpy(df["electrons"].values).to(self.device)
        e_dists = kernel * electrons[:, None, None]

        cols, rows = np.meshgrid(np.arange(self.size), np.arange(self.size))
        columns = (cols[None, :, :] + df["column"].values[:, None, None] - self.half_size).flatten()
        rows = (rows[None, :, :] + df["row"].values[:, None, None] - self.half_size).flatten()

        df = df.iloc[np.arange(len(df)).repeat(self.size * self.size)].reset_index(drop=True)
        df["column"] = columns
        df["row"] = rows
        if e_dists.is_cuda:
            e_dists = e_dists.cpu()
        df["charge"] = e_dists.numpy().flatten()
        return df


class CauchyKernelDiffuser(KernelDiffuser):
    def __init__(self, gamma, **kwargs):
        super().__init__(cauchy_gen(gamma), **kwargs)


class PowerFitDiffuser(Diffuser):
    CLUSTER_CACHE = {}
    COUNTS_CACHE = {}

    def __init__(self, a=4.2267, b=0.65,
                 cluster_file=os.path.join(os.path.dirname(__file__), "data/cluster_database_corrected.csv"),
                 **kwargs):
        super().__init__(**kwargs)

        self.a = a
        self.b = b

        try:
            self.cluster_database = self.CLUSTER_CACHE[cluster_file]
            self.cluster_counts = self.COUNTS_CACHE[cluster_file]
        except KeyError:
            cluster_db = pd.read_csv(cluster_file, converters={"hit_array": pd.eval})

            max_clusters = cluster_db["size"].value_counts().max()
            max_cs = 113  # Based on largest circular cluster generatable

            self.cluster_database: np.ndarray = np.zeros((max_cs+1, max_clusters, max_cs, 2))
            self.cluster_counts: np.ndarray = np.ones(max_cs+1)

            # Fill all sizes with the statically generated clusters, before overriding them with the database,
            # where the database has data.
            for size in range(1, 114):
                cluster = np.pad(self._create_cluster_circle(size), [(0, max_cs - size), (0, 0)],
                                 constant_values=np.iinfo(np.int32).max)
                self.cluster_database[size, 0] = np.expand_dims(cluster, axis=0)
            self.cluster_database[0, 0] = np.empty((max_cs, 2)).fill(np.iinfo(np.int32).max)

            for size in cluster_db["size"].unique():
                temp_db = []

                # Extract a subset of the database for our desired cluster size
                cluster_db_size = cluster_db[cluster_db["size"] == size]

                for cluster_entry in cluster_db_size.itertuples():
                    cluster = []
                    hit_arr = cluster_entry.hit_array
                    for y in range(10):
                        for x in range(10):
                            if hit_arr[y] & 2 ** x:
                                x_offset = math.floor(cluster_entry.x_mean + 0.5 - x)
                                y_offset = math.floor(cluster_entry.y_mean + 0.5 - y)
                                cluster.append((x_offset, y_offset))
                    temp_cluster = np.expand_dims(np.pad(np.array(cluster),
                                                         [(0, max_cs - size), (0, 0)],
                                                         constant_values=np.iinfo(np.int32).max), axis=0)
                    temp_db.append(temp_cluster)
                temp_stack = np.vstack(temp_db)

                self.cluster_database[size] = np.pad(temp_stack,
                                                     [(0, max_clusters - temp_stack.shape[0]), (0, 0), (0, 0)],
                                                     constant_values=np.iinfo(np.int32).max)
                self.cluster_counts[size] = temp_stack.shape[0]

            PowerFitDiffuser.CLUSTER_CACHE[cluster_file] = self.cluster_database
            PowerFitDiffuser.COUNTS_CACHE[cluster_file] = self.cluster_counts

    @staticmethod
    def _create_cluster_circle(size: int) -> np.ndarray:
        """
        cf. H. Pettersen's work (https://github.com/HelgeEgil/DigitalTrackingCalorimeterToolkit)
        Python version taken from G. Papp (https://github.com/pgpapp/pcT-ReadOut)

        :param size: The cluster size in pixels
        :return: List of x,y-tuples for all the activates pixels
        """
        circle_x = [0, 1, 0, -1, 0, 1, -1, -1, 1, 0, -2, 0, 2, 1, -2, -1, 2, -1, -2, 1, 2, -2, -2, 2, 2, 0, -3, 0,
                    3, -1, -3, 1, 3, 1, -3, -1, 3, 0, -4, 0, 4, 2, -3, -2, 3, -4, -2, -3, 2, 4, -1, -4, 1, 4, 1, 3,
                    -1, 3, 3, -3, -3, 4, 2, -4, -2, 4, -2, 2, -4, 5, 0, -5, 0, 5, -1, -5, 1, 5, 1, -5, -1, 6, 0, -6, 0,
                    6, -1, -6, 1, 6, 1, -6, -1, 5, -2, -5, 2, 5, 2, -5, -2, 4, -3, -4, 3, 3, -4, -3, 4, 7, 0, -7, 0]
        circle_y = [0, 0, -1, 0, 1, -1, -1, 1, 1, -2, 0, 2, 0, -2, -1, 2, 1, -2, 1, 2, -1, -2, 2, 2, -2, -3, 0, 3,
                    0, -3, 1, 3, -1, -3, -1, 3, 1, -4, 0, 4, 0, -3, -2, 3, 2, -1, -3, 2, 3, -1, -4, 1, 4, 1, -4, -2,
                    4, 3, -3, -3, 3, 2, -4, -2, 4, -2, -4, 4, 2, 0, -5, 0, 5, -1, -5, 1, 5, 1, -5, -1, 5, 0, -6, 0, 6,
                    -1, -6, 1, 6, 1, -6, -1, 6, -2, -5, 2, 5, 2, -5, -2, 5, -3, -4, 3, 4, -4, -3, 4, 3, 0, -7, 0, 7]
        pixels = []
        for i in range(min(size, len(circle_x))):
            pixels.append((circle_x[i], circle_y[i]))
        return np.array(pixels)

    def _get_cluster_of_size(self, size) -> np.ndarray:
        """
        Chooses a (random) cluster of the given size based on the cluster database specified in the constructor.

        :param size: Cluster size in number of active pixels.
        :return: List of tuples containing the offsets to apply to the original pixel in the form (x-offset, y-offset).
        """
        # Cap at max cluster size
        size = np.clip(size, 0, self.cluster_counts.shape[0] - 1).astype(np.int32)

        index = np.round(np.random.uniform(0, 1, (size.shape[0],)) * (self.cluster_counts[size] - 1)).astype(np.int32)

        return self.cluster_database[size, index]

    def _turn_hits_into_charges(self, df) -> pd.DataFrame:
        df["column"] = df["column"].astype(int)
        df["row"] = df["row"].astype(int)
        agg = {}
        for col in df.columns:
            agg[col] = "first"
        agg["edep"] = "sum"
        df = df.groupby(["eventID", "layer", "column", "row"], as_index=False).aggregate(agg)

        x_column_index = df.columns.get_loc("column")
        y_column_index = df.columns.get_loc("row")

        cs = self.a * (df["edep"] * 1000 / 25)**self.b + 0.5
        df["clusterSize"] = cs.astype(int)
        df = df[df["clusterSize"] > 0]
        original_types = df.dtypes

        hits = df[["column", "row"]].to_numpy()
        cluster_offsets = np.apply_along_axis(self._get_cluster_of_size, 0, df["clusterSize"])
        cluster = hits[:, None, :] + cluster_offsets

        mask_filter = cluster_offsets == np.iinfo(np.int32).max
        cluster = cluster[~mask_filter]

        cluster = cluster.reshape(-1, 2)

        tiled = np.repeat(df.to_numpy(), df["clusterSize"].values.clip(0, self.cluster_database.shape[0] -1), axis=0)
        tiled[:, (x_column_index, y_column_index)] = cluster

        df = pd.DataFrame(data=tiled, columns=df.columns).astype(original_types)
        df.drop(columns=["clusterSize"], inplace=True)

        df["charge"] = self.threshold

        df.reset_index(drop=True, inplace=True)
        return df


def diffuse_hits(df_hits, diffuser: Diffuser, diffused_path=None) -> pd.DataFrame:
    if diffused_path is not None and os.path.exists(diffused_path):
        try:
            return pd.DataFrame(np.load(diffused_path, allow_pickle=True))
        except Exception as ex:
            print("Failed to load file even though it exists, recreating...")
            print(ex)

    diffused = df_hits.groupby("eventID", as_index=False).apply(lambda x: diffuser.diffuse_hits(x))
    diffused = diffused.drop_duplicates(["eventID", "layer", "column", "row"])

    diffused["group_column"] = diffused["eventID"] * 100 + diffused["layer"]
    clustered = add_cluster_labels(diffused)

    df_hits = generate_cluster_df(clustered, consider_duplicates=False)
    # De-noising filters out 1-clusters
    df_hits = df_hits[df_hits["size"] > 1]

    df_hits["posX"] = df_hits["column_centroid"] * diffuser.px_w
    df_hits["posY"] = df_hits["row_centroid"] * diffuser.px_h
    layer_z = np.concatenate([[225.3415, 283.1415], 326.0815 + 5.5 * np.arange(41)])
    df_hits["posZ"] = layer_z[df_hits["layer"]]

    df_hits.drop(columns=["column", "row", "edep", "column_centroid", "row_centroid", "clusterSize",
                          "trackID", "cluster_id", "group_column"], inplace=True, errors="ignore")
    df_hits.reset_index(drop=True, inplace=True)

    if diffused_path is not None:
        np.save(diffused_path, df_hits.to_records(index=False))
    return df_hits
