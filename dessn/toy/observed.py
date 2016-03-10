from dessn.model.node import NodeObserved


class ObservedCounts(NodeObserved):
    def __init__(self, counts):
        super(ObservedCounts, self).__init__("Obs. Counts", "ocount", "$c_o$", counts)


class ObservedRedshift(NodeObserved):
    def __init__(self, redshifts, redshift_errors):
        super(ObservedRedshift, self).__init__("Obs. Redshift", ["oredshift", "oredshift_error"], ["$z_o$", r"$z_{o,{\rm err}}$"], [redshifts, redshift_errors])


class ObservedType(NodeObserved):
    def __init__(self, types):
        super(ObservedType, self).__init__("Obs. Type", "otype", "$T_o$", types)
