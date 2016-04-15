from dessn.model.parameter import ParameterObserved


class ObservedLightCurves(ParameterObserved):
    def __init__(self, lcs):
        super(ObservedLightCurves, self).__init__("olc", "$LC$", lcs, group="Obs. Light Curves")


class ObservedRedshift(ParameterObserved):
    def __init__(self, redshifts):
        super(ObservedRedshift, self).__init__("oredshift", "$z_o$", redshifts, group="Obs. Redshift")


class ObservedType(ParameterObserved):
    def __init__(self, types):
        super(ObservedType, self).__init__("otype", "$T_o$", types, group="Obs. Type")
