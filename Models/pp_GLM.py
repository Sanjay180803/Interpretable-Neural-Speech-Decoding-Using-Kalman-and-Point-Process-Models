# models/pp_glm.py

import numpy as np
from time import time
from typing import List, Tuple

from utils.preprocessing import load_data_pp


def fit_pp_glm(
    train_files: List[str],
    dt: float,
    stride: int = 5,
    max_samples: int = 200_000,
    max_files: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a Poisson GLM for each channel to predict envelope state [env, vel, acc].

    Returns
    -------
    alpha : (C,)          bias per channel
    beta  : (C, 3)        weights for [env, vel, acc] per channel
    """
    if max_files is None:
        max_files = len(train_files)

    X_all, Y_all = [], []
    n_ch = None

    print("=== PP_STEP_2: training_GLM ===")
    t0 = time()

    for i, path in enumerate(train_files[:max_files]):
        print(f"  file {i+1}/{max_files}: {path}")
        X_norm, Y, *_ = load_data_pp(path, stride=stride, max_bins=20_000)

        if n_ch is None:
            n_ch = Y.shape[1]
            print(f"  detected {n_ch} channels")

        env = X_norm[:, 0]
        vel = np.gradient(env, dt)
        acc = np.gradient(vel, dt)
        X_state = np.column_stack([env, vel, acc])

        X_all.append(X_state)
        Y_all.append(Y)

    X_cat = np.vstack(X_all)
    Y_cat = np.vstack(Y_all)

    if len(X_cat) > max_samples:
        idx = np.random.choice(len(X_cat), max_samples, replace=False)
        X_cat = X_cat[idx]
        Y_cat = Y_cat[idx]

    print(f"  training_samples = {len(X_cat)}")

    alpha = np.zeros(n_ch)
    beta = np.zeros((n_ch, 3))

    X_design = np.column_stack([np.ones(len(X_cat)), X_cat])

    for ch in range(n_ch):
        if (ch + 1) % 100 == 0 or ch == 0:
            print(f"    channel {ch+1}/{n_ch}")

        y = Y_cat[:, ch]

        if y.mean() < 0.05 or y.std() < 1e-6:
            alpha[ch] = -10.0
            continue

        log_y = np.log(y + 0.5)

        params = np.linalg.lstsq(X_design, log_y, rcond=None)[0]
        alpha[ch] = params[0]
        beta[ch] = params[1:]

    print(f"  done in {time() - t0:.1f}s")
    return alpha, beta


def train_state_dynamics_pp(
    train_files: List[str],
    dt: float,
    stride: int = 5,
    max_files: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn linear state dynamics x_t+1 = A x_t + w, cov(w) = Q for x=[env,vel,acc].
    """
    if max_files is None:
        max_files = len(train_files)

    X_prev_all, X_next_all = [], []
    print("=== PP_STEP_3: training_state_dynamics ===")
    t0 = time()

    for i, path in enumerate(train_files[:max_files]):
        print(f"  file {i+1}/{max_files}: {path}")
        X_norm, _, *_ = load_data_pp(path, stride=stride, max_bins=50_000)

        env = X_norm[:, 0]
        vel = np.gradient(env, dt)
        acc = np.gradient(vel, dt)
        X_state = np.column_stack([env, vel, acc])

        X_prev_all.append(X_state[:-1])
        X_next_all.append(X_state[1:])

    X_prev = np.vstack(X_prev_all)
    X_next = np.vstack(X_next_all)

    A = np.linalg.lstsq(X_prev, X_next, rcond=None)[0].T
    residuals = X_next - X_prev @ A.T
    Q = np.cov(residuals.T) * 2.0 + 1e-5 * np.eye(3)

    print(f"  done in {time() - t0:.1f}s")
    print("  Q_diag:", np.diag(Q))
    return A, Q


def decode_pp_ukf(
    X_norm_true: np.ndarray,
    Y_sel: np.ndarray,
    A: np.ndarray,
    Q: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Run UKF-style decoding where observation is Poisson GLM on x=[env,vel,acc].

    Returns
    -------
    x_filt : (T,3)   filtered state trajectory
    """
    from models.ukf import unscented_transform  # lazy import to avoid cycles

    T = len(Y_sel)
    n_ch = len(alpha)

    x_filt = np.zeros((T, 3))
    x_mean = np.array([X_norm_true[0, 0], 0.0, 0.0])
    P = 2.0 * np.eye(3)

    for t in range(T):
        sigma_pts, Wm, Wc = unscented_transform(x_mean, P)
        sigma_pred = sigma_pts @ A.T

        x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)

        P_pred = Q.copy()
        for i in range(len(sigma_pred)):
            d = sigma_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(d, d)
        P_pred += 1e-6 * np.eye(3)

        sigma_pts_up, Wm_up, Wc_up = unscented_transform(x_pred, P_pred)

        y_sigma = np.zeros((len(sigma_pts_up), n_ch))
        for i in range(len(sigma_pts_up)):
            log_lam = alpha + beta @ sigma_pts_up[i]
            log_lam = np.clip(log_lam, -5, 5)
            y_sigma[i] = np.exp(log_lam)

        y_pred = np.sum(Wm_up[:, None] * y_sigma, axis=0)

        P_yy = np.diag(y_pred + 1.0)
        for i in range(len(sigma_pts_up)):
            dy = y_sigma[i] - y_pred
            P_yy += Wc_up[i] * np.outer(dy, dy)

        P_xy = np.zeros((3, n_ch))
        for i in range(len(sigma_pts_up)):
            dx = sigma_pts_up[i] - x_pred
            dy = y_sigma[i] - y_pred
            P_xy += Wc_up[i] * np.outer(dx, dy)

        try:
            K = P_xy @ np.linalg.inv(P_yy)
            K = np.clip(K, -10, 10)
        except Exception:
            K = np.zeros((3, n_ch))

        innov = Y_sel[t] - y_pred
        x_mean = x_pred + K @ innov
        x_mean = np.clip(x_mean, [-8, -30, -70], [8, 30, 70])

        P = P_pred - K @ P_yy @ K.T
        P = (P + P.T) / 2 + 1e-5 * np.eye(3)

        x_filt[t] = x_mean

    return x_filt
