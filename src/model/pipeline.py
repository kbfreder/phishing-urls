import numpy as np
import pandas as pd
from collections import defaultdict
# import _feature_engineering as fe

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def gen_common_list(df, col, freq):
    if freq > 1:
        n = 0
    else:
        n = 1

    vcs = df[col].value_counts(n)
    return list(vcs[vcs > freq].index)


def gen_pipeline(col_list, proc_dict):
    pipes = []

    for col in col_list:
        i = 1
        steps = [('select', FeatureSelector(feature_names=[col]))]

        for step in proc_dict[col]:
            step_name = col + str(i)
            steps.append((step_name, step))
            i +=1

        pipes.append((col, Pipeline(steps=steps)))

    return pipes


class KFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self


class FeatureSelector(KFTransformer):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def transform(self, X, y=None):
        return X[self._feature_names]



class StrUpper(KFTransformer, BaseEstimator, TransformerMixin):
    '''Accepts single-column DataFrame. Converts column strings to uppercase
        Returns a Series'''

    def transform(self, X, *args):
        return X.iloc[:,0].str.upper()


def reshaper(X):
    return X.values.reshape(-1, 1)

Reshaper = FunctionTransformer(reshaper, validate=False)


class Consolidate(KFTransformer, BaseEstimator, TransformerMixin):
    '''Converts low-frequency values to 'OTHER'. Low-frequency = having absolute count below 'freq'.
        'freq' can be integer (absolute count) or float (relative percentage).
        Accepts single-column dataframe.
        Returns Series.'''

    def __init__(self, freq):
        self.freq = freq

    def fit(self, X, y=None):
        self.col = list(X.columns)[0]
        self.common_list = gen_common_list(X, self.col, self.freq)
        return self

    def transform(self, X, *args):
        return X[self.col].apply(lambda x: x if x in self.common_list else 'OTHER')


class FrequEncode(KFTransformer):
    '''Frequency-Encodes categorical column.
        Accepts single-column DataFrame.
        Returns Series.'''

    def transform(self, X, y=None):
        col = list(X.columns)[0]
        cts = X[col].value_counts()
        freq = cts / len(X)
        return X[col].map(freq)


class Imputer(KFTransformer, BaseEstimator, TransformerMixin):
    '''Accepts single-column DataFrame. Imputes null values with 'imp_val'. Returns a Series'''
    def __init__(self, imp_val):
        self.imp_val = imp_val

    def transform(self, X, *args):
        col = list(X.columns)
        res = X[col].fillna(value=self.imp_val)
        return res


def densify(X):
    return X.toarray()

Densify = FunctionTransformer(densify, validate=False)


class PrintDim(KFTransformer):
    '''Prints dimenstion of X. For trouble-shooting'''
    def transform(self, X):
        print(X.shape)


class PassThrough(KFTransformer):
    def transform(self, X):
        return X
