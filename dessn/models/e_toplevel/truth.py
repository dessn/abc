import numpy as np


def get_truths_labels_significance():
    # Name, Truth, Label, is_significant, min, max
    result = [
        ("Om", 0.3, r"$\Omega_m$", True, 0.1, 0.6),
        # ("w", -1.0, r"$w$", True, -1.5, -0.5),
        ("alpha", 0.14, r"$\alpha$", True, 0.0, 0.5),
        ("beta", 3.1, r"$\beta$", True, 0.0, 5.0),
        ("mean_MB", -19.365, r"$\langle M_B \rangle$", True, -19.6, -19.0),
        ("mean_x1", np.zeros(4), r"$\langle x_1^{%d} \rangle$", True, None, None),
        ("mean_c", np.zeros(4), r"$\langle c^{%d} \rangle$", True, None, None),
        ("sigma_MB", 0.1, r"$\sigma_{\rm m_B}$", True, 0.05, 0.3),
        ("sigma_x1", 0.5, r"$\sigma_{x_1}$", True, 0.1, 2.0),
        ("sigma_c", 0.1, r"$\sigma_c$", True, 0.05, 0.4),
        ("log_sigma_MB", np.log(0.1), r"$\log\sigma_{\rm m_B}$", True, -10.0, 1.0),
        ("log_sigma_x1", np.log(0.5), r"$\log\sigma_{x_1}$", True, -10.0, 1.0),
        ("log_sigma_c", np.log(0.1), r"$\log\sigma_c$", True, -10.0, 1.0),
        ("alpha_c", -5 * np.ones(4), r"$\alpha_c^{%d}$", True, None, None),
        ("dscale", 0.08, r"$\delta(0)$", False, -0.2, 0.2),
        ("dratio", 0.5, r"$\delta(\infty)/\delta(0)$", False, 0.0, 1.0),
        ("intrinsic_correlation", np.identity(3), r"$\rho$", False, None, None),
        ("calibration", np.zeros(8), r"$\delta \mathcal{Z}_%d$", True, None, None)
    ]
    return result