import numpy as np
import logging
import os


class Model(object):
    def __init__(self, filename):
        self.logger = logging.getLogger(__name__)
        self.filename = filename
        self.name = os.path.basename(filename)

