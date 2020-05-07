# pylint: disable=broad-except
"""Niseko dataset."""

import os
import json
import traceback

import numpy as np
import pandas as pd

from .meta_features import MetaFeatures
from .pipeline import NisekoPipeline
# from .pipeline import NisekoPipeline, NisekoPipelineRun


class NisekoDataset:
    """
    Dataset class.
    """

    def __init__(self, dataset_id, task_type, data_dir):
        self._dataset_id = dataset_id
        self._task_type = task_type
        self._data_dir = data_dir

        # load meta features
        meta_features = MetaFeatures.load(os.path.join(data_dir, 'meta_features', '{}.arff'.format(dataset_id)))
        self._meta_features = meta_features

    @property
    def dataset_id(self):
        return self._dataset_id

    @property
    def task_type(self):
        return self._task_type

    @property
    def meta_features(self):
        return self._meta_features

    @property
    def num_instances(self):
        return self.meta_features['NumberOfInstances'].value

    @property
    def num_features(self):
        return self.meta_features['NumberOfFeatures'].value

    def show_stats(self):
        print('Task Type: {}'.format(self.task_type))
        print(self.meta_features)

    def get_pipelines(self, order_by=None, num=None, clean=True, normalize=False):
        """
        Get all pipelines.
        :param order_by:
        :param num:
        :param clean:
        :param normalize:
        :return:
        """

        pipelines = self._get_pipelines()

        if clean:
            pipelines = self._clean_pipelines(pipelines)

        if normalize:
            self._normalize_pipelines(pipelines)

        if order_by is not None:
            pipelines = list(pipelines)
            if order_by == 'performance':
                pipelines = sorted(pipelines, key=lambda pipeline: pipeline.score, reverse=True)
            else:
                raise NotImplementedError()

        if num is not None:
            pipelines = pipelines[:num]

        return pipelines

    def get_pipeline_runs(self):
        for _, row in self._get_pipeline_runs().iterrows():
            yield NisekoPipelineRun(row)

    def _get_pipelines(self):
        """
        Get all pipelines.
        :return:
        """

        all_pipelines = []
        read_files = set()
        for root_dir, sub_dirs, _ in os.walk(self._data_dir):
            traces_path = os.path.join(root_dir, '{}.json'.format(self.dataset_id))
            if os.path.isfile(traces_path) and traces_path not in read_files:
                read_files.add(traces_path)
                try:
                    with open(traces_path, 'r') as f:
                        traces = json.load(f)
                    for pipeline in traces['pipelines']:
                        all_pipelines.append(NisekoPipeline.load_from_json(pipeline))
                except BaseException:
                    traceback.print_exc()

            for sub_dir in sub_dirs:
                pipelines_path = os.path.join(root_dir, sub_dir, '{}_pipelines.json'.format(self.dataset_id))
                if os.path.isfile(pipelines_path) and pipelines_path not in read_files:
                    read_files.add(pipelines_path)
                    try:
                        with open(pipelines_path, 'r') as f:
                            pipelines = json.load(f)
                        for pipeline in pipelines:
                            all_pipelines.append(NisekoPipeline.load_from_json(pipeline))
                    except BaseException:
                        pass

        return all_pipelines

    def _clean_pipelines(self, pipelines):
        """
        Get the cleaned pipelines.
        :param pipelines:
        :return:
        """

        if self.task_type == 'REGRESSION':
            scores = np.array(list(map(lambda pipeline: pipeline.score, pipelines)))
            scores = scores[~np.isnan(scores)]
            scores = scores[~np.isinf(scores)]
            score_median = np.median(scores)

        def pipeline_filter(pipeline):
            score = pipeline.score
            if np.isnan(score) or np.isinf(score):
                return False
            if self.task_type == 'REGRESSION':
                median_range = 10
                if score <= median_range * score_median:
                    return False

            return True

        cleaned_pipelines = list(filter(pipeline_filter, pipelines))
        return cleaned_pipelines

    def _normalize_pipelines(self, pipelines):
        """
        Normalize the scores of pipelines.
        :param pipelines:
        :return:
        """

        scores = np.array(list(map(lambda pipeline: pipeline.score, pipelines)))
        score_mean, score_std = np.mean(scores), np.std(scores)
        for pipeline in pipelines:
            pipeline.score = (pipeline.score - score_mean) / score_std

    def _get_pipeline_runs(self):
        """
        Get all pipeline runs.
        :return:
        """

        all_pipeline_runs = []
        read_files = set()
        for root_dir, sub_dirs, _ in os.walk(self.dumps_dir):
            pipeline_runs_path = os.path.join(root_dir, '{}_pipelines.csv'.format(self.dataset_id))
            if os.path.isfile(pipeline_runs_path) and pipeline_runs_path not in read_files:
                read_files.add(pipeline_runs_path)
                try:
                    with open(pipeline_runs_path, 'r') as f:
                        pipeline_runs = pd.read_csv(f)
                    all_pipeline_runs.append(pipeline_runs)
                except BaseException:
                    pass

            for sub_dir in sub_dirs:
                pipeline_runs_path = os.path.join(root_dir, sub_dir, '{}_pipelines.csv'.format(self.dataset_id))
                if os.path.isfile(pipeline_runs_path) and pipeline_runs_path not in read_files:
                    read_files.add(pipeline_runs_path)
                    try:
                        with open(pipeline_runs_path, 'r') as f:
                            pipeline_runs = pd.read_csv(f)
                        all_pipeline_runs.append(pipeline_runs)
                    except BaseException:
                        pass

        if all_pipeline_runs:
            return pd.concat(all_pipeline_runs, sort=False)
        return all_pipeline_runs
