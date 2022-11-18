import math
import numpy as np

from data_loaders import hdf5_utils
from edice.data.transforms import apply_transformation


class EpigenomeSliceDataset:

    """Just iterate over slices of the epigenome, in memory."""

    def __init__(
        self,
        signal_values,
        cell_ids,
        assay_ids,
        transform=None,
    ):
        self.X = apply_transformation(transform, signal_values)  # L, N
        self.cell_ids = np.asarray(cell_ids)
        self.assay_ids = np.asarray(assay_ids)

    def __getitem__(self, ix):
        return {
            "X": self.X[ix],
            "cell_ids": self.cellids,
            "assay_ids": self.assayids,
        }

    def __len__(self):
        return self.X.shape[0]
