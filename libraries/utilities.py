def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import collections
import statsmodels
import sklearn
import sys
import timeit
from math import *
from sklearn.feature_selection import *
from sklearn.preprocessing import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.cluster import *
from sklearn.linear_model import *
from hmmlearn import *
from sklearn.pipeline import *
from sklearn.grid_search import *
from sklearn.learning_curve import *
from sklearn.model_selection import *
from sklearn.manifold import *
from sklearn.metrics import *
from scipy.cluster.hierarchy import *
from matplotlib.collections import *
from sklearn.decomposition import *
from sklearn.cross_decomposition import *
from fastdtw import *


np.random.seed(42)
pd.set_option('display.max_columns', 51)
np.set_printoptions(precision=5, suppress=True)


#########################################################
def get_data(n_first_train=-1, n_first_test=-1):
    df_train = pd.read_csv('../data/numerai_training_data.csv', header=0)
    df_test = pd.read_csv('../data/numerai_tournament_data.csv', header=0)

    if n_first_train > 0:
        df_train = df_train.iloc[:n_first_train]

    if n_first_test > 0:
        df_test = df_test.iloc[:n_first_test]

    return df_train, df_test


#########################################################
def save_results(df_test, results):
    df = pd.DataFrame(data={'probability': results})
    joined = pd.DataFrame(df_test['t_id']).join(df)
    joined.to_csv("../predictions/predictions.csv", index=False)


#########################################################
def get_columns(df, with_target=True, n_first=-1):
    if with_target:
        columns = list(df.columns)
    else:
        columns = list(df.drop('target', axis=1).columns)

    if n_first > -1:
        columns = columns[:n_first]

    return columns
