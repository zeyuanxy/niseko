"""Niseko context."""

import os

import pandas as pd

from .dataset import NisekoDataset


class NisekoContext:
    """
    Niseko context class, this is the entry point for Niseko.
    """

    def __init__(self, data_dir):
        self._data_dir = data_dir
        self._datasets_df = pd.read_csv(os.path.join(data_dir, "datasets.csv"))

    def list_datasets(self, task_type=None):
        datasets = []
        for dataset_id in self._datasets_df['Dataset']:
            dataset_task_type = self._datasets_df[self._datasets_df['Dataset'] == dataset_id]['TaskType'].iloc[0]
            if task_type is None or dataset_task_type == task_type:
                datasets.append(self.get_dataset_by_id(dataset_id))
        return datasets

    def get_dataset_by_id(self, dataset_id):
        task_type = self._datasets_df[self._datasets_df['Dataset'] == dataset_id]['TaskType'].iloc[0]
        return NisekoDataset(dataset_id=dataset_id, task_type=task_type, data_dir=self._data_dir)


def get_context(dumps_dir):
    return NisekoContext(dumps_dir)
