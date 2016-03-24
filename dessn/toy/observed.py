from dessn.model.node import NodeObserved


class ObservedCounts(NodeObserved):
    def __init__(self, counts):
        super(ObservedCounts, self).__init__("ocount", "$c_o$", counts, group="Obs. Counts")


class ObservedRedshift(NodeObserved):
    def __init__(self, redshifts):
        super(ObservedRedshift, self).__init__("oredshift", "$z_o$", redshifts, group="Obs. Redshift")


class ObservedType(NodeObserved):
    def __init__(self, types):
        super(ObservedType, self).__init__("otype", "$T_o$", types, group="Obs. Type")
