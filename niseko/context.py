"""Niseko context."""
import os

import pandas as pd

from .dataset import NisekoDataset


class NisekoContext:

    def __init__(self, dumps_dir):
        self.datasets_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "datasets.csv"))
        self.dumps_dir = dumps_dir

    def list_datasets(self, task_type=None):
        datasets = []
        for dataset_id in self.datasets_df['Dataset']:
            dataset_task_type = self.datasets_df[self.datasets_df['Dataset'] == dataset_id]['TaskType'].iloc[0]
            if task_type is None or dataset_task_type == task_type:
                datasets.append(self.get_dataset_by_id(dataset_id))
        return datasets

    def get_dataset_by_id(self, dataset_id):
        task_type = self.datasets_df[self.datasets_df['Dataset'] == dataset_id]['TaskType'].iloc[0]
        return NisekoDataset(dataset_id=dataset_id, task_type=task_type, dumps_dir=self.dumps_dir)


def get_context(dumps_dir):
    return NisekoContext(dumps_dir)
