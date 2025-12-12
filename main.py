# main.py

import argparse
import numpy as np

from utils.preprocessing import make_splits, load_data_pp, load_data_kf, BASE_DT, STRIDE_DEFAULT
from models import pp_glm, kf as kf_mod, ukf as ukf_mod
from utils.metrics import compute_metrics, count_peaks_metric, compute_improved_ks
from utils.electrode_importance import compute_importances, plot_importances
from evaluation.topk_ablation import run_topk_ablation, plot_ablation


def train_all(data_dir: str):
    dt = BASE_DT * STRIDE_DEFAULT

    train_files, val_files, test_files = make_splits(data_dir)

    # --- channel selection on first train file ---
    X_norm_ex, Y_ex, *_ = load_data_pp(train_files[0], stride=STRIDE_DEFAULT, max_bins=50_000)
    n_ch_all = Y_ex.shape[1]
    scores = []
    for c in range(n_ch_all):
        corr_mat = np.corrcoef(X_norm_ex[:, 0], Y_ex[:, c])
        val = corr_mat[0, 1]
        scores.append(abs(val) if not np.isnan(val) else 0.0)
    ranked = np.argsort(scores)[::-1]
    selected_channels = ranked[:75]

    # --- PP ---
    alpha_pp, beta_pp = pp_glm.fit_pp_glm(train_files, dt=dt, stride=STRIDE_DEFAULT)
    A_pp, Q_pp = pp_glm.train_state_dynamics_pp(train_files, dt=dt, stride=STRIDE_DEFAULT)

    # --- KF ---
    A_lin, C_lin_full, sigma2_lin, Q_lin = kf_mod.train_kf_parameters(train_files)

    # --- UKF ---
    H_ukf, R_ukf = ukf_mod.fit_observation_model_ukf(train_files, dt=dt)
    W_env_ukf, W_vel_ukf, W_acc_ukf, Q_ukf = ukf_mod.train_state_dynamics_ukf(train_files, dt=dt)

    # --- basic evaluation on test set (envelope metrics only) ---
    results_pp, results_kf, results_ukf = [], [], []

    for path in test_files:
        print(f"\n=== Test file: {path} ===")
        # PP
        Xn_true, Y_raw, Xorig_true, Xm, Xs = load_data_pp(path, stride=STRIDE_DEFAULT, max_bins=30_000)
        Y_sel_pp = Y_raw[:, selected_channels]
        alpha_sel = alpha_pp[selected_channels]
        beta_sel = beta_pp[selected_channels]
        x_pp = pp_glm.decode_pp_ukf(Xn_true, Y_sel_pp, A_pp, Q_pp, alpha_sel, beta_sel, dt)
        dec_env_pp = x_pp[:, 0] * Xs + Xm
        actual_env = Xorig_true[:, 0]

        m_pp = compute_metrics(dec_env_pp, actual_env, allow_lag_correction=True)
        peak_pp = count_peaks_metric(m_pp["decoded_baseline_corrected"], actual_env)
        results_pp.append({**m_pp, **peak_pp})

        # KF
        Xn_kf, Yn_kf, Xorig_kf, Xm_kf, Xs_kf = load_data_kf(path, stride=STRIDE_DEFAULT, max_bins=30_000)
        Y_sel_kf = Yn_kf[:, selected_channels]
        C_sel = C_lin_full[selected_channels, :]
        _, x_kf, _ = kf_mod.kalman_filter_and_smoother(
            Xn_kf, Y_sel_kf, A_lin, C_sel, Q_lin, sigma2_lin
        )
        dec_env_kf = x_kf[:, 0] * Xs_kf + Xm_kf
        m_kf = compute_metrics(dec_env_kf, actual_env, allow_lag_correction=True)
        peak_kf = count_peaks_metric(m_kf["decoded_baseline_corrected"], actual_env)
        results_kf.append({**m_kf, **peak_kf})

        # UKF
        H_sel = H_ukf[selected_channels, :]
        R_sel = R_ukf[np.ix_(selected_channels, selected_channels)]
        x_ukf, _ = ukf_mod.decode_ukf(
            Xn_kf, Y_sel_kf, W_env_ukf, W_vel_ukf, W_acc_ukf, Q_ukf, H_sel, R_sel
        )
        dec_env_ukf = x_ukf[:, 0] * Xs_kf + Xm_kf
        m_ukf = compute_metrics(dec_env_ukf, actual_env, allow_lag_correction=True)
        peak_ukf = count_peaks_metric(m_ukf["decoded_baseline_corrected"], actual_env)
        results_ukf.append({**m_ukf, **peak_ukf})

        print(f"  PP_corr={m_pp['corr_baseline']:.3f}, KF_corr={m_kf['corr_baseline']:.3f}, UKF_corr={m_ukf['corr_baseline']:.3f}")

    # --- electrode importance ---
    glm_env_imp = np.abs(beta_pp[:, 0])
    glm_imp, kf_imp, ukf_imp, combined_imp = compute_importances(glm_env_imp, C_lin_full, H_ukf)
    plot_importances(glm_imp, kf_imp, ukf_imp, combined_imp)

    # --- example top-k ablation on first test file ---
    topk_vals, corr_pp, corr_kf, corr_ukf = run_topk_ablation(
        test_files[0],
        ranked,
        alpha_pp,
        beta_pp,
        A_pp,
        Q_pp,
        A_lin,
        C_lin_full,
        Q_lin,
        sigma2_lin,
        H_ukf,
        R_ukf,
        W_env_ukf,
        W_vel_ukf,
        W_acc_ukf,
        dt,
        stride=STRIDE_DEFAULT,
    )
    plot_ablation(topk_vals, corr_pp, corr_kf, corr_ukf)

    print("\nTraining_and_evaluation_complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with .mat sentence files")
    args = parser.parse_args()
    train_all(args.data_dir)
