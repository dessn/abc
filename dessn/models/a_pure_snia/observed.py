from dessn.framework.parameter import ParameterObserved


class ObservedLightCurves(ParameterObserved):
    def __init__(self, lcs):
        super(ObservedLightCurves, self).__init__("olc", r"$\hat{LC}$", lcs,
                                                  group="Obs. Light Curves")


class ObservedRedshift(ParameterObserved):
    def __init__(self, redshifts):
        super(ObservedRedshift, self).__init__("oredshift", r"$\hat{z}$", redshifts,
                                               group="Obs. Redshift")
