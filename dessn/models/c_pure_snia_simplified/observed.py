from dessn.framework.parameter import ParameterObserved


class ObservedMB(ParameterObserved):
    def __init__(self, mbs):
        super(ObservedMB, self).__init__("mb_o", r"$\hat{m_B}$", mbs, group="Lightcurve Fit")


class ObservedX1(ParameterObserved):
    def __init__(self, x1s):
        super(ObservedX1, self).__init__("x1_o", r"$\hat{x_1}$", x1s, group="Lightcurve Fit")


class ObservedC(ParameterObserved):
    def __init__(self, cs):
        super(ObservedC, self).__init__("c_o", r"$\hat{c}$", cs, group="Lightcurve Fit")


class ObservedInvCovariance(ParameterObserved):
    def __init__(self, inv_covs):
        super(ObservedInvCovariance, self).__init__("inv_cov", r"$\hat{C}^{-1}$", inv_covs, group="Lightcurve Fit")


class ObservedCovariance(ParameterObserved):
    def __init__(self, covs):
        super(ObservedCovariance, self).__init__("cov", r"$\hat{C}$", covs, group="Lightcurve Fit")


class ObservedRedshift(ParameterObserved):
    def __init__(self, redshifts):
        super(ObservedRedshift, self).__init__("oredshift", r"$\hat{z_o}$", redshifts, group="Obs. Redshift")
