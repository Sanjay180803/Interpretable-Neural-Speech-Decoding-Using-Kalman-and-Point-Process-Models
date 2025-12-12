# models/kf.py

import numpy as np
from typing import List, Tuple

from utils.preprocessing import load_data_kf


def train_kf_parameters(
    train_files: List[str],
    stride: int = 5,
    max_bins: int = 50_000,
    q_scale: float = 5.0,
    sigma2_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Learn linear state model:
        x_t = A x_{t-1} + w,  y_t = C x_t + v
    where x is 1-D (envelope) and y_t is all channels.

    Returns
    -------
    A_lin      : (1,1)
    C_lin_full : (C,1)
    sigma2_lin : scalar observation variance
    """
    XtX = 0.0
    XtY = None
    sum_xx = 0.0
    sum_xy = 0.0

    for path in train_files:
        X_norm, Y_norm, *_ = load_data_kf(path, stride=stride, max_bins=max_bins)

        if XtY is None:
            XtY = np.zeros((1, Y_norm.shape[1]))

        XtX += float(X_norm.T @ X_norm)
        XtY += X_norm.T @ Y_norm

        x_prev = X_norm[:-1, :]
        x_next = X_norm[1:, :]

        sum_xx += float(x_prev.T @ x_prev)
        sum_xy += float(x_prev.T @ x_next)

    C_lin_full = XtY.T / (XtX + 1e-12)
    A_lin = np.array([[sum_xy / (sum_xx + 1e-12)]])

    # process noise Q and observation noise sigma2
    sum_d2 = 0.0
    count_d = 0
    sigma2_sum = 0.0
    sigma2_count = 0

    for path in train_files:
        X_norm, Y_norm, *_ = load_data_kf(path, stride=stride, max_bins=max_bins)

        x_prev = X_norm[:-1, :]
        x_next = X_norm[1:, :]
        diff = x_next - A_lin[0, 0] * x_prev

        sum_d2 += float(np.sum(diff ** 2))
        count_d += diff.size

        pred_Y = X_norm @ C_lin_full.T
        residual = Y_norm - pred_Y
        sigma2_sum += float(np.mean(residual ** 2))
        sigma2_count += 1

    Q_base = sum_d2 / (count_d + 1e-12)
    sigma2_base = sigma2_sum / (sigma2_count + 1e-12)

    Q_lin = np.array([[Q_base * q_scale]])
    sigma2_lin = sigma2_base * sigma2_scale

    print(f"[KF] A = {A_lin[0,0]:.4f}")
    print(f"[KF] Q_base={Q_base:.6f}, Q_scaled={Q_lin[0,0]:.6f}")
    print(f"[KF] sigma2_base={sigma2_base:.6f}, sigma2_scaled={sigma2_lin:.6f}")

    return A_lin, C_lin_full, sigma2_lin, Q_lin


def kalman_filter_and_smoother(
    X: np.ndarray,
    Y_sel: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    sigma2: float,
    use_smoother: bool = False,
    init_p: float = 1.0,
    track_gain_steps: int = 0,
):
    """
    1-D linear KF on envelope with multi-channel observations.
    """
    T = X.shape[0]
    Ct_flat = C.flatten()
    CtC = float(Ct_flat @ Ct_flat)

    x_filt = np.zeros_like(X)
    P = np.zeros(T)
    P_pred_arr = np.zeros(T)

    x_curr = float(Y_sel[0] @ Ct_flat / (CtC + 1e-6))
    P_curr = init_p
    gain_norms = []

    for t in range(T):
        x_pred = A[0, 0] * x_curr
        P_pred = A[0, 0] * P_curr * A[0, 0] + Q[0, 0]
        P_pred_arr[t] = P_pred

        denom = CtC * P_pred + sigma2
        K_t = (P_pred / (denom + 1e-12)) * Ct_flat

        if t < track_gain_steps:
            gain_norms.append(float(np.linalg.norm(K_t)))

        pred_Y_t = x_pred * Ct_flat
        innovation = Y_sel[t] - pred_Y_t

        x_curr = x_pred + float(K_t @ innovation)
        P_curr = (1.0 - float(Ct_flat @ K_t)) * P_pred

        x_filt[t, 0] = x_curr
        P[t] = P_curr

    if not use_smoother:
        return x_filt, x_filt, gain_norms

    # RTS smoother
    x_smooth = np.zeros_like(X)
    x_smooth[-1, 0] = x_filt[-1, 0]
    P_smooth = np.zeros_like(P)
    P_smooth[-1] = P[-1]

    for t in range(T - 2, -1, -1):
        P_t = P[t]
        P_pred_next = P_pred_arr[t + 1]
        x_pred_next = A[0, 0] * x_filt[t, 0]

        if P_pred_next <= 1e-12:
            J_t = 0.0
        else:
            J_t = (P_t * A[0, 0]) / P_pred_next

        x_smooth[t, 0] = x_filt[t, 0] + J_t * (x_smooth[t + 1, 0] - x_pred_next)
        P_smooth[t] = P_t + J_t * (P_smooth[t + 1] - P_pred_next) * J_t

    return x_filt, x_smooth, gain_norms
