# utils/electrode_importance.py

import numpy as np
import matplotlib.pyplot as plt


def compute_importances(glm_env_importance, C_lin_full, H_ukf):
    """
    Combine PP, KF, UKF weights into a single importance score.

    Returns
    -------
    glm_importance, kf_importance, ukf_importance, combined_importance
    """
    glm_imp = np.abs(glm_env_importance)
    kf_imp = np.abs(C_lin_full[:, 0])
    ukf_imp = np.linalg.norm(H_ukf, axis=1)

    combined = 0.4 * glm_imp + 0.3 * kf_imp + 0.3 * ukf_imp
    return glm_imp, kf_imp, ukf_imp, combined


def plot_importances(glm_imp, kf_imp, ukf_imp, combined_imp):
    plt.figure(figsize=(10, 4))
    plt.stem(glm_imp)
    plt.title("PP GLM Envelope Weight Magnitude per Channel")
    plt.xlabel("Channel index")
    plt.ylabel("|β_env|")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.stem(kf_imp)
    plt.title("KF Linear Weight Magnitude per Channel")
    plt.xlabel("Channel index")
    plt.ylabel("|C|")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.stem(ukf_imp)
    plt.title("UKF Observation Norm per Channel (‖H_row‖)")
    plt.xlabel("Channel index")
    plt.ylabel("‖H_row‖")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.stem(combined_imp)
    plt.title("Combined Electrode Importance (PP + KF + UKF)")
    plt.xlabel("Channel index")
    plt.ylabel("Importance score")
    plt.tight_layout()
    plt.show()
