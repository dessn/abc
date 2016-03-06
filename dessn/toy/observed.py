from dessn.model.node import NodeObserved


class ObservedCounts(NodeObserved):
    def __init__(self, counts):
        super(ObservedCounts, self).__init__("Obs. Counts", "ocount", "$c_0$", counts)


class ObservedRedshift(NodeObserved):
    def __init__(self, redshifts):
        super(ObservedRedshift, self).__init__("Obs. Redshift", "oredshift", "$z_0$", redshifts)


class ObservedType(NodeObserved):
    def __init__(self, types):
        super(ObservedType, self).__init__("Obs. Type", "otype", "$T_0$", types)
