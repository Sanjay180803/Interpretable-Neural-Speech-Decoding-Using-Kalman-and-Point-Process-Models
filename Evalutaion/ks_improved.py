# evaluation/ks_improved.py

import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import improved_ks_plot_on_axis


def average_ks_plots(xs_list, actual_cdf_list, decoded_cdf_list):
    """
    Build an averaged KS plot from multiple per-file CDFs.
    """
    if not xs_list:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x_min = min(xs.min() for xs in xs_list)
    x_max = max(xs.max() for xs in xs_list)
    xs_common = np.linspace(x_min, x_max, 500)

    act_interp_all, dec_interp_all = [], []

    for xs, act_cdf, dec_cdf in zip(xs_list, actual_cdf_list, decoded_cdf_list):
        act_interp = np.interp(xs_common, xs, act_cdf)
        dec_interp = np.interp(xs_common, xs, dec_cdf)
        act_interp_all.append(act_interp)
        dec_interp_all.append(dec_interp)
        ax.plot(xs_common, act_interp, color="blue", alpha=0.15)
        ax.plot(xs_common, dec_interp, color="red", alpha=0.15)

    act_mean = np.mean(act_interp_all, axis=0)
    dec_mean = np.mean(dec_interp_all, axis=0)

    ax.plot(xs_common, act_mean, "b-", linewidth=2.5, label="Mean Actual")
    ax.plot(xs_common, dec_mean, "r-", linewidth=2.5, label="Mean Decoded")

    diff_mean = np.abs(act_mean - dec_mean)
    idx_mean = np.argmax(diff_mean)
    ks_mean_dist = diff_mean[idx_mean]
    xs_ks_mean = xs_common[idx_mean]

    ax.vlines(xs_ks_mean,
              min(act_mean[idx_mean], dec_mean[idx_mean]),
              max(act_mean[idx_mean], dec_mean[idx_mean]),
              colors="green", linestyles="--", linewidth=2,
              label=f"Mean KS ≈ {ks_mean_dist:.3f}")

    ax.set_title("Average Improved KS Plot")
    ax.set_xlabel("log(envelope+1)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
