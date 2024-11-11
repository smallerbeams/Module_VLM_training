import datetime
import json
import os
import time
import random
import torch
from pathlib import Path
from typing import List

import webdataset as wds
from process_block.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)

from process_block.Dataloader.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)

from torch.utils.data import DataLoader,DistributedSampler


class Minigptv2_Mutli_Dataloader:

    def __init__(self, datasets, is_trains, num_workers,model_conf=None):
        self.datasets = datasets
        self._dataloaders = None
        self.num_workers = num_workers
        self.is_trains = is_trains
        self.use_distributed = model_conf['use_distributed']
    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler

                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):

            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )

            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            print("dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline).")


            batch_sizes = {dataset_name: self.datasets[dataset_name]['batch_size']
                           for dataset_name in self.datasets.keys()}
            total_mutli_datasets = {dataset_name: {'train' : self.datasets[dataset_name]['train']}
                           for dataset_name in self.datasets.keys()}
            
            datasets, batch_sizes = reorg_datasets_by_split(total_mutli_datasets, batch_sizes)
            self.datasets = datasets

            # self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
            
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        print("Only a single wds.DataPipeline dataset, no __len__ attribute.")

                if num_records >= 0:
                    print("Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name)
                        )
            
            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            batch_sizes = [batch_sizes[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            print("batch sizes", batch_sizes)

            collate_fns = []
            
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])

                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader
    
    @property
    def train_splits(self):
        train_splits = ['train']

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            print("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

def reorg_datasets_by_split(datasets, batch_sizes):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()
    reorg_batch_sizes = dict()

    # reorganize by split
    for dataset_name, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():

            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
                reorg_batch_sizes[split_name] = [batch_sizes[dataset_name]]
            else:
                reorg_datasets[split_name].append(dataset_split)
                reorg_batch_sizes[split_name].append(batch_sizes[dataset_name])

    return reorg_datasets, reorg_batch_sizes

class ChainDataset(wds.DataPipeline):
    r"""Dataset for chaining multiple :class:`DataPipeline` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: List[wds.DataPipeline]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prob = []
        self.names = []
        for dataset in self.datasets:
            if hasattr(dataset, 'name'):
                self.names.append(dataset.name)
            else:
                self.names.append('Unknown')
            if hasattr(dataset, 'sample_ratio'):
                self.prob.append(dataset.sample_ratio)
            else:
                self.prob.append(1)
                print("One of the datapipeline doesn't define ratio and set to 1 automatically.")

    def __iter__(self):
        datastreams = [iter(dataset) for dataset in self.datasets]
        while True:
            select_datastream = random.choices(datastreams, weights=self.prob, k=1)[0]
            yield next(select_datastream)