from dessn.framework.models.approx_model import ApproximateModel


class FullModel(ApproximateModel):

    def __init__(self, num_supernova, file="full.stan"):
        super().__init__(num_supernova, file=file)

    def get_data(self, simulation, cosmology_index, add_zs=None):
        return super().get_data(simulation, cosmology_index, add_zs=300)
