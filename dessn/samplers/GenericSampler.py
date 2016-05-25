import os


class Sampler(object):
    """ A generic sampler

    Parameters
    ----------
    temp_dir : str, optional
        The directory in which to store results, such as the output chains
    """
    def __init__(self, temp_dir=None):
        self.temp_dir = temp_dir
        if self.temp_dir is not None:
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)

    def get_chain(self):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()