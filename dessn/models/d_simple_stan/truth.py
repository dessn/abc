import numpy as np


def get_truths_labels_significance():
    # Name, Truth, Label, is_significant, min, max
    result = [
        ("Om", 0.3, r"$\Omega_m$", True, 0.1, 0.6),
        # ("w", -1.0, r"$w$", True, -1.5, -0.5),
        ("alpha", 0.1, r"$\alpha$", True, 0, 0.5),
        ("beta", 3.0, r"$\beta$", True, 0, 5),
        ("mean_MB", -19.3, r"$\langle M_B \rangle$", True, -19.6, -19),
        ("mean_x1", 0.2, r"$\langle x_1 \rangle$", True, -0.5, 0.5),
        ("mean_c", 0.1, r"$\langle c \rangle$", True, -0.2, 0.2),
        ("sigma_MB", 0.1, r"$\sigma_{\rm m_B}$", True, 0.05, 0.3),
        ("sigma_x1", 0.5, r"$\sigma_{x_1}$", True, 0.1, 2.0),
        ("sigma_c", 0.1, r"$\sigma_c$", True, 0.05, 0.4),
        ("alpha_MB", 0.0, r"$\alpha_{\rm m_B}$", True, -1, 1),
        ("alpha_x1", 0.0, r"$\alpha_{x_1}$", True, -1, 1),
        ("alpha_c", 0.0, r"$\alpha_c$", True, -1, 1),
        ("dscale", 0.08, r"$\delta(0)$", False, -0.2, 0.2),
        ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$", False, 0.0, 1.0),
        ("intrinsic_correlation", np.identity(3), r"$\rho$", False, None, None),
        ("calibration", np.zeros(4), r"$\delta \mathcal{Z}_%d$", True, None, None)
    ]
    return result