# -*- coding: utf-8 -*-
"""Classes of feature mapping for model type B
"""

import numpy as np
# import matplotlib.pyplot as plt
# import random
# from arch import arch_model
import pandas as pd
import math
# import pmdarima as pm
# from pmdarima import model_selection
# import os
# import dis
# import statistics
# from sklearn import metrics
# import sklearn
from tsfresh import extract_features

from statsmodels.tsa.seasonal import seasonal_decompose

# import itertools
# import functools
import warnings
from builtins import range
# from collections import defaultdict


from numpy.linalg import LinAlgError
# from scipy.signal import cwt, find_peaks_cwt, ricker, welch
# from scipy.stats import linregress
# from statsmodels.tools.sm_exceptions import MissingDataError

with warnings.catch_warnings():
    # Ignore warnings of the patsy package
    warnings.simplefilter("ignore", DeprecationWarning)

    from statsmodels.tsa.ar_model import AR
# from statsmodels.tsa.stattools import acf, adfuller, pacf

from hurst import compute_Hc

class Window:
    """ The  class for rolling window feature mapping.
    The mapping converts the original timeseries X into a matrix. 
    The matrix consists of rows of sliding windows of original X. 
    """

    def __init__(self,  window = 100):
        self.window = window
        self.detector = None
    def convert(self, X):
        n = self.window
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
        else:
            for i in range(n):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[n-1:]
        return df

class tf_Stat:
    '''statisitc feature extraction using the tf_feature package. 
    It calculates 763 features in total so it might be over complicated for some models. 
    Recommend to use for methods like Isolation Forest which randomly picks a feature
    and then perform the classification. To use for other distance-based model like KNN,
    LOF, CBLOF, etc, first train to pass a function that give weights to individual features so that
    inconsequential features won't cloud the important ones (mean, variance, kurtosis, etc).

    '''
    def __init__(self,  window = 100, step = 25):
        self.window = window
        self.step = step
        self.detector = None
    def convert(self, X):
        window = self.window
        step = self.step
        pos = math.ceil(window/2)
        #step <= window

        length = X.shape[0]

        Xd = pd.DataFrame(X)
        Xd.columns = pd.Index(['x'], dtype='object')
        Xd['id'] = 1
        Xd['time'] = Xd.index
        
        test = np.array(extract_features(Xd.iloc[0+pos-math.ceil(window/2):0+pos + math.floor(window/2)], column_id="id", column_sort="time", column_kind=None, column_value=None).fillna(0))
        M = np.zeros((length - window, test.shape[1]+1 ))

        
        i = 0
        while i + window <= M.shape[0]:
            M[i:i+step, 0]= X[pos + i: pos + i + step]
            vector = np.array(extract_features(Xd.iloc[i+pos-math.ceil(window/2):i+pos + math.floor(window/2)], column_id="id", column_sort="time", column_kind=None, column_value=None).fillna(0))

            M[i:i+step, 1:] = vector
            i+= step
        num = M.shape[0]
        if i <  num:
            M[i: num, 0]= X[pos + i: pos + num]
            M[i: num, 1:] = np.array(extract_features(Xd.iloc[i+pos-math.ceil(window/2):], column_id="id", column_sort="time", column_kind=None, column_value=None).fillna(0))
        return M

class Stat:
    '''statisitc feature extraction. 
    Features include [mean, variance, skewness, kurtosis, autocorrelation, maximum, 
    minimum, entropy, seasonality, hurst component, AR coef]

    '''
    def __init__(self,  window = 100, data_step = 10, param = [{"coeff": 0, "k": 5}], lag = 1, freq = 720):
        self.window = window
        self.data_step = data_step
        self.detector = None
        self.param = param
        self.lag = lag 
        self.freq =freq
        if data_step > int(window/2):
            raise ValueError('value step shoudm\'t be greater than half of the window')
        
        
    def convert(self, X):
        freq = self.freq
        n = self.window
        data_step = self.data_step
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
            raise ValueError('window lenght is set to zero')
        else:
            for i in range(n):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[n:]
            df2 = pd.concat(L[:data_step], axis = 1)

        
        
        df = df.reset_index()
        #value 
        x0 = df2[math.ceil(n/2) : - math.floor(n/2)].reset_index()
        #mean 
        x1 = (df.mean(axis=1))
        #variance 
        x2 = df.var(axis=1)
        #AR-coef
        self.ar_function = lambda x: self.ar_coefficient(x)
        x3 = df.apply(self.ar_function, axis =1, result_type='expand'  )
        #autocorrelation
        self.auto_function = lambda x: self.autocorrelation(x)
        x4 = df.apply(self.auto_function, axis =1, result_type='expand'  )
        #kurtosis
        x5 = (df.kurtosis(axis=1))
        #skewness
        x6 = (df.skew(axis=1))
        #maximum
        x7 = (df.max(axis=1))
        #minimum
        x8 = (df.min(axis=1))
        #entropy
        self.entropy_function = lambda x: self.sample_entropy(x)
        x9 = df.apply(self.entropy_function, axis =1, result_type='expand')
        
        #seasonality
        result = seasonal_decompose(X, model='additive', freq = freq, extrapolate_trend='freq')
        #seasonal
        x10 = pd.Series(np.array(result.seasonal[math.ceil(n/2) : - math.floor(n/2)]))
        #trend 
        x11 = pd.Series(np.array(result.trend[math.ceil(n/2) : - math.floor(n/2)]))
        #resid 
        x12 = pd.Series(np.array(result.resid[math.ceil(n/2) : - math.floor(n/2)]))
        
        #Hurst component
        self.hurst_function = lambda x: self.hurst_f(x)
        x13 = df.apply(self.hurst_function, axis =1, result_type='expand')
        
        L = [x0, x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12, x13]
        M = pd.concat(L, axis = 1)
        M = M.drop(columns=['index'])

        return M
    def ar_coefficient(self, x):
        """
        This feature calculator fits the unconditional maximum likelihood
        of an autoregressive AR(k) process.
        The k parameter is the maximum lag of the process

        .. math::

            X_{t}=\\varphi_0 +\\sum _{{i=1}}^{k}\\varphi_{i}X_{{t-i}}+\\varepsilon_{t}

        For the configurations from param which should contain the maxlag "k" and such an AR process is calculated. Then
        the coefficients :math:`\\varphi_{i}` whose index :math:`i` contained from "coeff" are returned.

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param param: contains dictionaries {"coeff": x, "k": y} with x,y int
        :type param: list
        :return x: the different feature values
        :return type: pandas.Series
        """
        calculated_ar_params = {}
        param = self.param
        x_as_list = list(x)

        res = {}

        for parameter_combination in param:
            k = parameter_combination["k"]
            p = parameter_combination["coeff"]

            column_name = "coeff_{}__k_{}".format(p, k)

            if k not in calculated_ar_params:
                try:
                    calculated_AR = AR(x_as_list)
                    calculated_ar_params[k] = calculated_AR.fit(maxlag=k, solver="mle").params
                except (LinAlgError, ValueError):
                    calculated_ar_params[k] = [np.NaN] * k

            mod = calculated_ar_params[k]

            if p <= k:
                try:
                    res[column_name] = mod[p]
                except IndexError:
                    res[column_name] = 0
            else:
                res[column_name] = np.NaN

        L = [(key, value) for key, value in res.items()]
        L0 = []
        for item in L:
            L0.append(item[1])
        return L0

    def autocorrelation(self, x):
        """
        Calculates the autocorrelation of the specified lag, according to the formula [1]

        .. math::

            \\frac{1}{(n-l)\\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

        where :math:`n` is the length of the time series :math:`X_i`, :math:`\\sigma^2` its variance and :math:`\\mu` its
        mean. `l` denotes the lag.

        .. rubric:: References

        [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
        :param lag: the lag
        :type lag: int
        :return: the value of this feature
        :return type: float
        """
        lag = self.lag
        # This is important: If a series is passed, the product below is calculated
        # based on the index, which corresponds to squaring the series.
        if isinstance(x, pd.Series):
            x = x.values
        if len(x) < lag:
            return np.nan
        # Slice the relevant subseries based on the lag
        y1 = x[:(len(x) - lag)]
        y2 = x[lag:]
        # Subtract the mean of the whole series x
        x_mean = np.mean(x)
        # The result is sometimes referred to as "covariation"
        sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
        # Return the normalized unbiased covariance
        v = np.var(x)
        if np.isclose(v, 0):
            return np.NaN
        else:
            return sum_product / ((len(x) - lag) * v)
    def _into_subchunks(self, x, subchunk_length, every_n=1):
        """
        Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

        For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

            0  2  4
            1  3  5
            2  4  6

        with the settings subchunk_length = 3 and every_n = 2
        """
        len_x = len(x)

        assert subchunk_length > 1
        assert every_n > 0

        # how often can we shift a window of size subchunk_length over the input?
        num_shifts = (len_x - subchunk_length) // every_n + 1
        shift_starts = every_n * np.arange(num_shifts)
        indices = np.arange(subchunk_length)

        indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
        return np.asarray(x)[indexer]
    def sample_entropy(self, x):
        """
        Calculate and return sample entropy of x.

        .. rubric:: References

        |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
        |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract

        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray

        :return: the value of this feature
        :return type: float
        """
        x = np.array(x)

        # if one of the values is NaN, we can not compute anything meaningful
        if np.isnan(x).any():
            return np.nan

        m = 2  # common value for m, according to wikipedia...
        tolerance = 0.2 * np.std(x)  # 0.2 is a common value for r, according to wikipedia...

        # Split time series and save all templates of length m
        # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
        xm = self._into_subchunks(x, m)

        # Now calculate the maximum distance between each of those pairs
        #   np.abs(xmi - xm).max(axis=1)
        # and check how many are below the tolerance.
        # For speed reasons, we are not doing this in a nested for loop,
        # but with numpy magic.
        # Example:
        # if x = [1, 2, 3]
        # then xm = [[1, 2], [2, 3]]
        # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
        # and from [2, 3] => [[1, 1], [0, 0]]
        # taking the abs and max gives us:
        # [0, 1] and [1, 0]
        # as the diagonal elements are always 0, we substract 1.
        B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

        # Similar for computing A
        xmp1 = self._into_subchunks(x, m + 1)

        A = np.sum([np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1])

        # Return SampEn
        return -np.log(A / B)
    def hurst_f(self, x):
        H,c, M = compute_Hc(x)
        return [H, c]