from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RandForest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from validation.CV import *
import enum


embedded = [Lasso, ElasticNet, Ridge, RandForest]
wrappers = [LogisticRegression, LDA, SVC, Tree, GaussianNB]

help_desc = """HELP"""


class PipelineType(enum.Enum):
    Wrapper = 1
    Embedded = 2
    Filter = 3


class InvalidSelectorException(Exception):
    pass

class InvalidClassifierException(Exception):
    pass

class MissingClassifierException(Exception):
    def __init__(self):
        Exception.__init__(self, "well, that rather badly didnt it?")

class MissingValidator(Exception):
    pass

class MissingDataset(Exception):
    pass


class Pipeline:

    def __validate_selector(self):
        if self.selector is None:
            self.pipelineType = PipelineType.Embedded
        elif type(self.selector) is RFE:
            self.pipelineType = PipelineType.Wrapper
        elif type(self.selector) is GenericUnivariateSelect:
            self.pipelineType = PipelineType.Filter
        else:
            raise InvalidSelectorException

    def __validate_classifier(self):
        if self.classifier is None:
            raise MissingClassifierException
        elif self.pipelineType is PipelineType.Embedded and type(self.classifier) in embedded:
            pass
        elif type(self.classifier) in wrappers:
            pass
        else:
            raise InvalidClassifierException

    def __validate_validator(self):
        if self.validator is None:
            raise MissingValidator

    def __validate_dataset(self):
        if self.X is None or self.y is None:
            raise MissingDataset

    def __init__(self, selector, classifier, validator, X, y):
        self.classifier = classifier
        self.selector = selector
        self.validator = validator
        self.X = X
        self.y = y

        self.__validate_selector()

        self.__validate_classifier()

        self.__validate_validator()

        self.__validate_dataset()

    def run(self):

        if self.pipelineType is PipelineType.Embedded:

            return validate(self.X, self.y, self.classifier, self.validator)

        if self.pipelineType is PipelineType.Filter or PipelineType.Wrapper:

            new_X = self.selector.fit_transform(self.X, self.y)

            return validate(new_X, self.y, self.classifier, self.validator)

    @staticmethod
    def generate_random_pipeline(X, y):
        return Pipeline(None, Lasso(alpha=0.1), KFold(n_splits=3, shuffle=True), X, y)
