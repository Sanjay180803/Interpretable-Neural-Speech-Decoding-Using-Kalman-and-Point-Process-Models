# models/ukf.py

import numpy as np
from typing import List, Tuple

from utils.preprocessing import load_data_kf


def unscented_transform(x_mean, P, alpha_ut=1e-3, beta_ut=2.0, kappa_ut=0.0):
    n = len(x_mean)
    lam = alpha_ut**2 * (n + kappa_ut) - n

    Wm = np.zeros(2 * n + 1)
    Wc = np.zeros(2 * n + 1)

    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha_ut**2 + beta_ut)

    Wm[1:] = 1.0 / (2 * (n + lam))
    Wc[1:] = Wm[1:]

    sigma_pts = np.zeros((2 * n + 1, n))
    sigma_pts[0] = x_mean

    try:
        sqrt_P = np.linalg.cholesky((n + lam) * P)
    except Exception:
        eigval, eigvec = np.linalg.eigh(P)
        eigval = np.maximum(eigval, 1e-10)
        sqrt_P = eigvec @ np.diag(np.sqrt(eigval * (n + lam)))

    for i in range(n):
        sigma_pts[i + 1] = x_mean + sqrt_P[:, i]
        sigma_pts[n + i + 1] = x_mean - sqrt_P[:, i]

    return sigma_pts, Wm, Wc


def fit_observation_model_ukf(
    train_files: List[str],
    dt: float,
    stride: int = 5,
    max_samples: int = 200_000,
    max_files: int | None = None,
):
    """
    Learn linear observation y = H x + v in the 3-D state space.
    """
    if max_files is None:
        max_files = len(train_files)

    X_all, Y_all = [], []
    n_ch = None

    print("[UKF] training_observation_model")
    for i, path in enumerate(train_files[:max_files]):
        print(f"  file {i+1}/{max_files}: {path}")
        X_norm, Y_norm, *_ = load_data_kf(path, stride=stride, max_bins=20_000)
        if n_ch is None:
            n_ch = Y_norm.shape[1]
        env = X_norm[:, 0]
        vel = np.gradient(env, dt)
        acc = np.gradient(vel, dt)
        X_all.append(np.column_stack([env, vel, acc]))
        Y_all.append(Y_norm)

    X_cat = np.vstack(X_all)
    Y_cat = np.vstack(Y_all)

    if len(X_cat) > max_samples:
        idx = np.random.choice(len(X_cat), max_samples, replace=False)
        X_cat = X_cat[idx]
        Y_cat = Y_cat[idx]

    H = np.linalg.lstsq(X_cat, Y_cat, rcond=None)[0].T
    Y_pred = X_cat @ H.T
    residuals = Y_cat - Y_pred
    chan_vars = np.var(residuals, axis=0)
    R = np.diag(chan_vars + 1e-6)

    print(f"[UKF] H shape = {H.shape}")
    print(f"[UKF] mean|H_env|={np.mean(np.abs(H[:,0])):.4f}")
    return H, R


def train_state_dynamics_ukf(
    train_files: List[str],
    dt: float,
    stride: int = 5,
    max_files: int | None = None,
):
    """
    Fit nonlinear dynamics x_{t+1} = f(x_t) with polynomial + bias features.
    """
    if max_files is None:
        max_files = len(train_files)

    X_feat_all, next_env_all, next_vel_all, next_acc_all = [], [], [], []

    print("[UKF] training_state_dynamics")
    for i, path in enumerate(train_files[:max_files]):
        print(f"  file {i+1}/{max_files}: {path}")
        X_norm, *_ = load_data_kf(path, stride=stride, max_bins=50_000)
        env = X_norm[:, 0]
        vel = np.gradient(env, dt)
        acc = np.gradient(vel, dt)

        env_sq = env**2
        feats = np.column_stack([env, env_sq, vel, acc, np.ones(len(env))])

        X_feat_all.append(feats[:-1])
        next_env_all.append(env[1:])
        next_vel_all.append(vel[1:])
        next_acc_all.append(acc[1:])

    X_feat = np.vstack(X_feat_all)
    env_next = np.hstack(next_env_all)
    vel_next = np.hstack(next_vel_all)
    acc_next = np.hstack(next_acc_all)

    W_env = np.linalg.lstsq(X_feat, env_next, rcond=None)[0]
    W_vel = np.linalg.lstsq(X_feat, vel_next, rcond=None)[0]
    W_acc = np.linalg.lstsq(X_feat, acc_next, rcond=None)[0]

    res_env = env_next - X_feat @ W_env
    res_vel = vel_next - X_feat @ W_vel
    res_acc = acc_next - X_feat @ W_acc

    residuals = np.vstack([res_env, res_vel, res_acc]).T
    Q_ukf = np.cov(residuals.T) + 1e-6 * np.eye(3)

    print("[UKF] Q_diag:", np.diag(Q_ukf))
    return W_env, W_vel, W_acc, Q_ukf


def f_state(x, W_env, W_vel, W_acc):
    env, vel, acc = x
    feats = np.array([env, env**2, vel, acc, 1.0])
    env_next = W_env @ feats
    vel_next = W_vel @ feats
    acc_next = W_acc @ feats
    return np.array([env_next, vel_next, acc_next])


def decode_ukf(
    X_true: np.ndarray,
    Y_sel: np.ndarray,
    W_env,
    W_vel,
    W_acc,
    Q_ukf,
    H,
    R,
):
    """
    Nonlinear UKF decoding with learned f(x) and H.
    """
    T = len(Y_sel)
    n_state = 3
    n_ch = Y_sel.shape[1]

    x_filt = np.zeros((T, n_state))
    x_mean = np.array([X_true[0, 0], 0.0, 0.0], dtype=np.float64)
    P = 0.1 * np.eye(n_state)
    I3 = np.eye(n_state)

    gain_norms = []

    for t in range(T):
        sigma_pts, Wm, Wc = unscented_transform(x_mean, P)
        sigma_pred = np.zeros_like(sigma_pts)
        for i in range(sigma_pts.shape[0]):
            sigma_pred[i] = f_state(sigma_pts[i], W_env, W_vel, W_acc)

        x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)

        P_pred = Q_ukf.copy()
        for i in range(sigma_pred.shape[0]):
            dx = sigma_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(dx, dx)
        P_pred += 1e-6 * I3

        y_sigma = np.zeros((sigma_pred.shape[0], n_ch))
        for i in range(sigma_pred.shape[0]):
            y_sigma[i] = H @ sigma_pred[i]

        y_pred = np.sum(Wm[:, None] * y_sigma, axis=0)

        P_yy = R.copy()
        for i in range(y_sigma.shape[0]):
            dy = y_sigma[i] - y_pred
            P_yy += Wc[i] * np.outer(dy, dy)

        P_xy = np.zeros((n_state, n_ch))
        for i in range(sigma_pred.shape[0]):
            dx = sigma_pred[i] - x_pred
            dy = y_sigma[i] - y_pred
            P_xy += Wc[i] * np.outer(dx, dy)

        try:
            K = P_xy @ np.linalg.inv(P_yy)
        except np.linalg.LinAlgError:
            K = np.zeros((n_state, n_ch))

        gain_norms.append(float(np.linalg.norm(K)))
        innov = Y_sel[t] - y_pred
        x_mean = x_pred + K @ innov
        x_mean[0] = np.clip(x_mean[0], -6, 6)

        P = P_pred - K @ P_yy @ K.T
        P = (P + P.T) / 2 + 1e-6 * I3

        x_filt[t] = x_mean

    return x_filt, gain_norms
