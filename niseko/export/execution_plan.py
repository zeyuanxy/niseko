from abc import ABC, abstractmethod
from collections import namedtuple

import pandas as pd


class BasePrimitive(ABC):
    """
    The base class for primitives, here we adopt the design of D3M primitives
    """

    def __init__(self):
        pass

    @abstractmethod
    def set_training_data(self, inputs, outputs=None):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def produce(self, inputs):
        pass


CallResult = namedtuple('CallResult', ['value'])
CallResult.__new__.__defaults__ = (None,) * len(CallResult._fields)


class SKLearnPrimitive(BasePrimitive):

    def __init__(self, primitive):
        self.primitive = primitive

    def set_training_data(self, inputs, outputs=None):
        self._inputs = inputs
        self._outputs = outputs

    def fit(self):
        if self._outputs is None:
            self.primitive.fit(self._inputs)
        else:
            self.primitive.fit(self._inputs, self._outputs)

        self._inputs = None
        self._outputs = None

    def produce(self, inputs):
        if getattr(self.primitive, "predict", None):
            return CallResult(pd.DataFrame(self.primitive.predict(inputs)))
        else:
            return CallResult(pd.DataFrame(self.primitive.transform(inputs)))


class XGBoostPrimitive(SKLearnPrimitive):

    def set_training_data(self, inputs, outputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.values

        self._inputs = inputs
        self._outputs = outputs

    def produce(self, inputs):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.values

        return CallResult(pd.DataFrame(self.primitive.predict(inputs)))


LOGICAL_TO_PHYSICAL_TABLE = {
    # data
    'DatasetToDataFrame': 'blinded.execution.primitive.contribs.common.DatasetToDataFrame',
    'ExtractColumnsBySemanticTypes': 'blinded.execution.primitive.contribs.common.ExtractColumnsBySemanticTypes',
    'HorizontalConcat': 'blinded.execution.primitive.contribs.common.HorizontalConcat',
    'ColumnParser': 'blinded.execution.primitive.contribs.common.ColumnParser',
    'NDArrayToDataFrame': 'blinded.execution.primitive.contribs.common.NDArrayToDataFrame',
    'Denormalize': 'blinded.execution.primitive.contribs.common.Denormalize',
    'DataFrameToList': 'blinded.execution.primitive.contribs.common.DataFrameToList',

    # feature
    'Imputer': 'sklearn.preprocessing.Imputer',
    'MinMaxScaler': 'sklearn.preprocessing.MinMaxScaler',
    'StandardScaler': 'sklearn.preprocessing.StandardScaler',
    'RobustScaler': 'sklearn.preprocessing.RobustScaler',
    'LabelEncoder': 'sklearn.preprocessing.LabelEncoder',
    'OneHotEncoder': 'autosklearn.pipeline.implementations.OneHotEncoder.OneHotEncoder',
    'PCA': 'sklearn.decomposition.PCA',
    'KernelPCA': 'sklearn.decomposition.KernelPCA',
    'TruncatedSVD': 'sklearn.decomposition.TruncatedSVD',
    'FastICA': 'sklearn.decomposition.FastICA',
    'PolynomialFeatures': 'sklearn.preprocessing.PolynomialFeatures',
    'SelectPercentile': 'sklearn.feature_selection.SelectPercentile',
    'GenericUnivariateSelect': 'sklearn.feature_selection.GenericUnivariateSelect',
    # 'SelectKBest': 'sklearn.feature_selection.SelectKBest',  # missing
    'VarianceThreshold': 'sklearn.feature_selection.VarianceThreshold',
    'FeatureAgglomeration': 'sklearn.cluster.FeatureAgglomeration',
    'RBFSampler': 'sklearn.kernel_approximation.RBFSampler',
    'Normalizer': 'sklearn.preprocessing.Normalizer',

    # classification
    'SVC': 'sklearn.svm.SVC',
    'LinearSVC': 'sklearn.svm.LinearSVC',
    'LogisticRegression': 'sklearn.linear_model.LogisticRegression',
    'SGDClassifier': 'sklearn.linear_model.SGDClassifier',
    'RandomForestClassifier': 'sklearn.ensemble.RandomForestClassifier',
    'GaussianNB': 'sklearn.naive_bayes.GaussianNB',
    'AdaBoostClassifier': 'sklearn.ensemble.AdaBoostClassifier',  # missing
    'KNeighborsClassifier': 'sklearn.neighbors.KNeighborsClassifier',
    'BaggingClassifier': 'sklearn.ensemble.BaggingClassifier',
    'ExtraTreesClassifier': 'sklearn.ensemble.ExtraTreesClassifier',
    'GradientBoostingClassifier': 'sklearn.ensemble.GradientBoostingClassifier',
    'XGradientBoostingClassifier': 'xgboost.XGBClassifier',
    'LinearDiscriminantAnalysis': 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
    'QuadraticDiscriminantAnalysis': 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis',
    'DecisionTreeClassifier': 'sklearn.tree.DecisionTreeClassifier',
    'LGBMClassifier': 'lightgbm.LGBMClassifier',

    # regression
    'SVR': 'sklearn.svm.SVR',
    'LinearSVR': 'sklearn.svm.LinearSVR',
    'LinearRegression': 'sklearn.linear_model.LinearRegression',  # missing
    'Ridge': 'sklearn.linear_model.Ridge',
    'SGDRegressor': 'sklearn.linear_model.SGDRegressor',
    'RandomForestRegressor': 'sklearn.ensemble.RandomForestRegressor',
    'GaussianProcessRegressor': 'sklearn.gaussian_process.GaussianProcessRegressor',
    'AdaBoostRegressor': 'sklearn.ensemble.AdaBoostRegressor',  # missing
    'KNeighborsRegressor': 'sklearn.neighbors.KNeighborsRegressor',
    'BaggingRegressor': 'sklearn.ensemble.BaggingRegressor',  # missing
    'ExtraTreesRegressor': 'sklearn.ensemble.ExtraTreesRegressor',
    'GradientBoostingRegressor': 'sklearn.ensemble.GradientBoostingRegressor',
    'XGradientBoostingRegressor': 'xgboost.XGBRegressor',
    'ARDRegression': 'sklearn.linear_model.ARDRegression',
    'DecisionTreeRegressor': 'sklearn.tree.DecisionTreeRegressor',
    'LGBMRegressor': 'lightgbm.LGBMRegressor',
    'RuleFit': 'rulefit.RuleFit',
}


def get_primitive_path(primitive_name):
    return LOGICAL_TO_PHYSICAL_TABLE[primitive_name]
