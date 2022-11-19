"""
    Classes and utils for holding and working with dataset metadata 
        (track names and cell/assay id mapping) and data (signal values)

    Currently supports data stored in HDF5 file (HDF5Dataset)
"""
import os
from pathlib import Path

import h5py
import numpy as np

from edice.constants import DATA_DIR
from edice.data.datasets import EpigenomeSliceDataset
from edice.data import hdf5_utils
from edice.data.annotations import IntervalAnnotation


class HDF5DataModule:

    def __init__(
        self,
        filepath,
        idmap=None,
        tracks=None, 
        splits=None,
        total_bins=None,
        data_dir=None,
        chromosome=None,
        name=None,
        transformation=None,
    ):
        """
        fixed_inputs: whether to generate a fixed size vector representing all possible tracks,
            in a fixed order, or to only return those tracks to be used as supports 
            at a given datapoint (along with cell, assay ids to identify the tracks)
        """
        data_dir = Path(data_dir or DATA_DIR)
        self.hdf5_file = str(data_dir/filepath)
        self.chromosome = chromosome
        self.name = name or self.__class__.name
        self.transformation = transformation

        if tracks is None:
            with h5py.File(self.hdf5_file, 'r') as h5f:
                tracks = [t.decode() for t in h5f['track_names'][:]]
                self.total_bins = total_bins or h5f['targets'].shape[0]

        super().__init__(tracks=tracks, idmap=idmap, splits=splits)

    def load_gaps(self):
        self.gaps = IntervalAnnotation.from_gap(os.path.join(DATA_DIR, self.gap_file))

    def load_blacklist(self):
        self.blacklist = IntervalAnnotation.from_bed(os.path.join(DATA_DIR, self.blacklist_file), extra_cols=["reason"])

    def _get_bins_with_gaps(self):
        if not hasattr(self, "gaps"):
            self.load_gaps()
        bins_with_gaps = self.gaps.get_chrom_annotated_bins(self.chromosome, bin_size=self.track_resolution)
        return [b for b in bins_with_gaps if b < self.total_bins]

    def _get_blacklist_bins(self):
        if not hasattr(self, "blacklist"):
            self.load_blacklist()
        blacklist_bins = self.blacklist.get_chrom_annotated_bins(self.chromosome, bin_size=self.track_resolution)
        return [b for b in blacklist_bins if b < self.total_bins]

    def _get_bin_mask(
        self,
        start_bin=0,
        total_bins=None,
        exclude_gaps=False,
        exclude_blacklist=False,
    ):
        total_bins = total_bins or self.total_bins
        # first get a mask covering all bins from 0 to start_bin + total_bins
        # then slice to get mask for the bins start_bin:start_bin + total_bins
        mask = np.ones(start_bin+total_bins, dtype=bool)
        if exclude_gaps:
            assert self.chromosome is not None and self.gap_file is not None,\
                ("Exclude gaps can only be specified on a chromosome"
                 "dataset with a gap_file attribute")
            bins_with_gaps = self._get_bins_with_gaps()
            # Set mask elements corresponding to gap bins to False
            mask[bins_with_gaps] = False
            print(f"Excluding {mask.shape[0]} - {mask.sum()} bins with gaps (starting from 0)")
        
        if exclude_blacklist:
            assert self.chromosome is not None and self.blacklist_file is not None,\
                ("Exclude blacklist can only be specified on a chromosome"
                 "dataset with a blacklist_file attribute")
            blacklist_bins = self._get_blacklist_bins()
            mask[blacklist_bins] = False
            print(f"Excluding {mask.shape[0]} - {mask.sum()} bins with gaps or blacklist (starting from 0)")

        mask = mask[start_bin:]  # slice back to (total_bins,)
        print(f"Keeping {mask.sum()} bins of {mask.shape[0]} (starting from {start_bin})")
        return mask

    def load_tracks(
        self,
        tracks,
        start_bin=0,
        total_bins=None,
        exclude_gaps=False,
        exclude_blacklist=False,
    ):
        mask = self._get_bin_mask(
            start_bin=start_bin,
            total_bins=total_bins,
            exclude_gaps=exclude_gaps,
            exclude_blacklist=exclude_blacklist,
        )
        assert mask.shape[0] == (total_bins or self.total_bins)
        print("Loading data from h5 file", flush=True)
        track_ids = np.asarray([self.track2id[t] for t in tracks])
        return hdf5_utils.load_tracks_from_hdf5_by_id(
            self.hdf5_file, track_ids, start_bin,
            mask.shape[0], mask)

    def prepare_data(
        self,
        tracks,
        exclude_gaps=False,
        exclude_blacklist=False,
    ):
        """
        support_tracks, target_tracks: lists of tracks
        return_track_ids: return track ids rather than loading values
        for tracks (for use with ValHDF5Generator)

        supports, cell_ids, assay_ids, _ , _ , _ = self.prepare_data(
            train_tracks, exclude_gaps=exclude_gaps, exclude_blacklist=exclude_blacklist
        )
        """
        cell_ids = [
            self.cell2id[self.get_track_cell(t)] 
            for t in tracks
        ]
        assay_ids = [
            self.assay2id[self.get_track_assay(t)]
            for t in tracks
        ]

        tracks = self.load_tracks(
            tracks,
            exclude_gaps=exclude_gaps,
            exclude_blacklist=exclude_blacklist,
        )

        return tracks, cell_ids, assay_ids
