import argparse
import os
import glob
import json
import numpy as np

from util.plot_utils import plot_pvals_mm, apply_style


def merge_dicts(dicts):
    result = {}
    for k in dicts[0].keys():
        values = [d[k] for d in dicts]
        if isinstance(dicts[0][k], dict):
            result[k] = merge_dicts(values)
        else:
            result[k] = values
    return result


def aggregate_dict(d):
    result = {}
    for k in d.keys():
        if isinstance(d[k], dict):
            result[k] = aggregate_dict(d[k])
        else:
            values = np.array(d[k])
            if len(values.shape) == 1:
                result[f"{k}_mean"] = np.mean(d[k])
                result[f"{k}_std"] = np.std(d[k])
            else:
                result[k] = values
    return result


def compare_configs(cfg1, cfg2):
    ignore_keys = ["seed", "device", "purge_workdir", "workdir"]
    cfg1_filtered = {k:v for k,v in cfg1.items() if k not in ignore_keys}
    cfg2_filtered = {k:v for k,v in cfg2.items() if k not in ignore_keys}
    return cfg1_filtered == cfg2_filtered


def verify_and_load_config(directories: list[str]):
    cfg = None
    for pattern in directories:
        for path in glob.glob(pattern):
            with open(os.path.join(path, "config.json")) as f:
                new_cfg = json.load(f)
                if cfg is None:
                    cfg = new_cfg
                if not compare_configs(cfg, new_cfg):
                    raise ValueError(f"Configs do not match between all runs")
    return cfg


def load_results(directories: list[str]):
    result_dicts = []
    for pattern in directories:
        for path in glob.glob(pattern):
            with open(os.path.join(path, "results.json")) as f:
                result_dicts.append(json.load(f))
    return result_dicts


def build_error_table(res):
    s_r_mae_mean = res["single_task_r"]["mae_0_mean"]
    s_r_mae_std = res["single_task_r"]["mae_0_std"]
    s_r_rmse_mean = res["single_task_r"]["rmse_0_mean"]
    s_r_rmse_std = res["single_task_r"]["rmse_0_std"]
    s_z_mae_mean = res["single_task_z"]["mae_0_mean"]
    s_z_mae_std = res["single_task_z"]["mae_0_std"]
    s_z_rmse_mean = res["single_task_z"]["rmse_0_mean"]
    s_z_rmse_std = res["single_task_z"]["rmse_0_std"]
    m_r_mae_mean = res["multitask_weighted_sum"]["mae_0_mean"]
    m_r_mae_std = res["multitask_weighted_sum"]["mae_0_std"]
    m_r_rmse_mean = res["multitask_weighted_sum"]["rmse_0_mean"]
    m_r_rmse_std = res["multitask_weighted_sum"]["rmse_0_std"]
    m_z_mae_mean = res["multitask_weighted_sum"]["mae_1_mean"]
    m_z_mae_std = res["multitask_weighted_sum"]["mae_1_std"]
    m_z_rmse_mean = res["multitask_weighted_sum"]["rmse_1_mean"]
    m_z_rmse_std = res["multitask_weighted_sum"]["rmse_1_std"]
    h_r_mae_mean = res["multitask_homoscedastic"]["mae_0_mean"]
    h_r_mae_std = res["multitask_homoscedastic"]["mae_0_std"]
    h_r_rmse_mean = res["multitask_homoscedastic"]["rmse_0_mean"]
    h_r_rmse_std = res["multitask_homoscedastic"]["rmse_0_std"]
    h_z_mae_mean = res["multitask_homoscedastic"]["mae_1_mean"]
    h_z_mae_std = res["multitask_homoscedastic"]["mae_1_std"]
    h_z_rmse_mean = res["multitask_homoscedastic"]["rmse_1_mean"]
    h_z_rmse_std = res["multitask_homoscedastic"]["rmse_1_std"]
    table_code = f"        Single task " \
                 f"& ${s_r_mae_mean:.3f} \\pm {s_r_mae_std:.3f}$ " \
                 f"& ${s_z_mae_mean:.3f} \\pm {s_z_mae_std:.3f}$ " \
                 f"& ${s_r_rmse_mean:.3f} \\pm {s_r_rmse_std:.3f}$ " \
                 f"& ${s_z_rmse_mean:.3f} \\pm {s_z_rmse_std:.3f}$\\\\\n" \
                 f"        Weighted sum " \
                 f"& ${m_r_mae_mean:.3f} \\pm {m_r_mae_std:.3f}$ " \
                 f"& ${m_z_mae_mean:.3f} \\pm {m_z_mae_std:.3f}$ " \
                 f"& ${m_r_rmse_mean:.3f} \\pm {m_r_rmse_std:.3f}$ " \
                 f"& ${m_z_rmse_mean:.3f} \\pm {m_z_rmse_std:.3f}$\\\\\n" \
                 f"        Homoscedastic " \
                 f"& ${h_r_mae_mean:.3f} \\pm {h_r_mae_std:.3f}$ " \
                 f"& ${h_z_mae_mean:.3f} \\pm {h_z_mae_std:.3f}$ " \
                 f"& ${h_r_rmse_mean:.3f} \\pm {h_r_rmse_std:.3f}$ " \
                 f"& ${h_z_rmse_mean:.3f} \\pm {h_z_rmse_std:.3f}$\\\\\n"
    print("MAE/RMSE table")
    print(table_code)


def build_transfer_error_table(res):
    m_r_mae_mean = res["multitask_weighted_sum"]["other"]["mae_0_mean"]
    m_r_mae_std = res["multitask_weighted_sum"]["other"]["mae_0_std"]
    m_z_mae_mean = res["multitask_weighted_sum"]["other"]["mae_1_mean"]
    m_z_mae_std = res["multitask_weighted_sum"]["other"]["mae_1_std"]
    m_r_rmse_mean = res["multitask_weighted_sum"]["other"]["rmse_0_mean"]
    m_r_rmse_std = res["multitask_weighted_sum"]["other"]["rmse_0_std"]
    m_z_rmse_mean = res["multitask_weighted_sum"]["other"]["rmse_1_mean"]
    m_z_rmse_std = res["multitask_weighted_sum"]["other"]["rmse_1_std"]
    table_code = f"${m_r_mae_mean:.3f} \\pm {m_r_mae_std:.3f}$ " \
                 f"& ${m_z_mae_mean:.3f} \\pm {m_z_mae_std:.3f}$ " \
                 f"& ${m_r_rmse_mean:.3f} \\pm {m_r_rmse_std:.3f}$ " \
                 f"& ${m_z_rmse_mean:.3f} \\pm {m_z_rmse_std:.3f}$\\\\"
    print("MAE/RMSE values on 'other' data")
    print(table_code)
    print("Rejection rate:", res["multitask_weighted_sum"]["other"]["rr_1_calib_mean"], "+-", res["multitask_weighted_sum"]["other"]["rr_1_calib_std"])


def plot_pvalues(cfg, res):
    spots = np.arange(cfg["ttest"]["spots_min"], cfg["ttest"]["spots_max"] + 1, cfg["ttest"]["spots_step"])
    pvalues = {k: v["pvalues"] for k, v in res.items()}
    filter_models = ["multitask_weighted_sum", "multitask_homoscedastic", "single_task_z"]
    plot_pvals_mm(spots, pvalues, 1, filter_models, output_path="ttest_spot_1mm.pdf")
    plot_pvals_mm(spots, pvalues, 2, filter_models, output_path="ttest_spot_2mm.pdf")
    plot_pvals_mm(spots, pvalues, 3, filter_models, output_path="ttest_spot_3mm.pdf")
    plot_pvals_mm(spots, pvalues, 4, filter_models, output_path="ttest_spot_4mm.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", metavar="directories", type=str, nargs="+")
    args = parser.parse_args()

    apply_style()

    config = verify_and_load_config(args.directories)
    results = load_results(args.directories)
    aggregated_results = aggregate_dict(merge_dicts(results))

    build_error_table(aggregated_results)
    build_transfer_error_table(aggregated_results)

    plot_pvalues(config, aggregated_results)
