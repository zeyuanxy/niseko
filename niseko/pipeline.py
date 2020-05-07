# pylint: disable=broad-except
"""Niseko pipeline."""

import json

import pandas as pd

from .export import convert_pipeline_to_script


class NisekoPipelineStep:
    """
    Pipeline step.
    """

    def __init__(self, step):
        self._primitive = step['primitive']['name']
        self._hyperparameters = step['primitive'].get('humanReadableParameters', {})
        self._inputs = step['inputs']

    @property
    def primitive(self):
        return self._primitive

    @property
    def hyperparameters(self):
        return self._hyperparameters

    @property
    def inputs(self):
        return self._inputs

    def get_hyperparameters_vector(self):
        return list(map(lambda item: item[1], sorted(map(lambda item: (item[0], float(item[1])),
                                                         self.hyperparameters.items()))))


class NisekoPipelineRun:
    """
    This class stores the information of a pipeline run.
    """

    def __init__(self, row):
        self._row = row

    def has_model(self, model):
        return self._row[model]

    @property
    def training_time(self):
        return self._row['train_time']

    @property
    def time(self):
        return self._row['train_time'] + self._row['validation_time']

    @property
    def sample_size(self):
        return self._row['sample_size']

    @property
    def error(self):
        return self._row['validation_error']


class NisekoPipeline:
    """
    Pipeline class.
    """

    def __init__(self):
        self._pipeline_runs = None

    @property
    def pipeline_id(self):
        return self._pipeline_id

    @pipeline_id.setter
    def pipeline_id(self, pipeline_id):
        self._pipeline_id = pipeline_id

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def primitives(self):
        return self._primitives

    @primitives.setter
    def primitives(self, primitives):
        self._primitives = primitives

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        self._steps = steps

    @property
    def pipeline_runs(self):
        """
        Get all pipeline runs.
        :return:
        """

        if self._pipeline_runs is None:
            pipeline_runs = []
            try:
                progression = pd.DataFrame.from_dict(json.loads(self.pipeline_json['metrics']['progression']))
                for _, pipeline_run in progression.iterrows():
                    pipeline_runs.append(PipelineRun(pipeline_run))
            except BaseException:
                pass
            self.pipeline_runs = pipeline_runs
            self.pipeline_json = None

        return self._pipeline_runs

    @pipeline_runs.setter
    def pipeline_runs(self, pipeline_runs):
        self._pipeline_runs = pipeline_runs

    @property
    def pipeline_json(self):
        return self._pipeline_json

    @pipeline_json.setter
    def pipeline_json(self, pipeline_json):
        self._pipeline_json = pipeline_json

    def to_script(self, output_path=None):
        # this code only works with the AutoML runtime package installed
        # otherwise it may fail

        script = convert_pipeline_to_script(self.steps)
        if output_path is None:
            print(script)
        else:
            with open(output_path, 'w') as f:
                f.write(script)

    def get_step(self, name):
        for step in self.steps:
            if step.primitive == name:
                return step

        return None

    @staticmethod
    def load_from_json(pipeline_json):
        """
        Load a pipeline from a json file.
        :param pipeline_json:
        :return:
        """

        pipeline = NisekoPipeline()
        pipeline.score = pipeline_json['metrics']['score']
        pipeline.model = pipeline_json['tags']['model']

        primitives = []
        steps = []
        for step in pipeline_json['pipeline']['steps']:
            primitives.append(step['primitive']['name'])
            steps.append(PipelineStep(step))
        pipeline.primitives = primitives
        pipeline.steps = steps
        pipeline.pipeline_json = pipeline_json

        return pipeline
