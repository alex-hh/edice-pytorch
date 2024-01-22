import math
import numpy as np

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
            "cell_ids": self.cell_ids,
            "assay_ids": self.assay_ids,
        }

    def __len__(self):
        return self.X.shape[0]


class EpigenomeSliceWithTargets:

    """Dataset partitioning train and test tracks.
    """

    def __init__(
        self,
        train_values,
        train_cell_ids,
        train_assay_ids,
        test_values,
        test_cell_ids,
        test_assay_ids,
        transform=None,
    ):
        self.X = apply_transformation(transform, train_values)  # L, N
        self.train_cell_ids = np.asarray(train_cell_ids)
        self.train_assay_ids = np.asarray(train_assay_ids)
        self.val_X = apply_transformation(transform, test_values)
        assert self.val_X.shape[0] == self.X.shape[0]
        self.test_cell_ids = np.asarray(test_cell_ids)
        self.test_assay_ids = np.asarray(test_assay_ids)

    def __getitem__(self, ix):
        return {
            "X": self.X[ix],
            "targets": self.val_X[ix],
            "cell_ids": self.train_cell_ids,
            "assay_ids": self.train_assay_ids,
            "target_cell_ids": self.test_cell_ids,
            "target_assay_ids": self.test_assay_ids,
        }

    def __len__(self):
        return self.X.shape[0]
