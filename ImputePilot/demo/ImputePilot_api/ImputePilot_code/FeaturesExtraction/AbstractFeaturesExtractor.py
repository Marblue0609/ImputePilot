"""
ImputePilot: A Demo System for Stable Imputation Model Selection in Time Series Data Repair
Zhejiang University
***
AbstractFeaturesExtractor.py
@author: zhexinjin@zju.edu.cn
"""

import abc
import os
from os.path import isfile, normpath as normp, dirname, abspath

from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils
from ImputePilot_api.ImputePilot_code.Utils.SingletonClass import SingletonClass

class AbstractFeaturesExtractor(SingletonClass, metaclass=abc.ABCMeta):
    """
    Abstract features extracting class used to extract time series features and handle those features.
    """
    
    _CURRENT_DIR = dirname(abspath(__file__))
    FEATURES_DIR = normp(os.path.join(_CURRENT_DIR, 'features/'))
    FEATURES_APPENDIX = '_features.csv'

    # create necessary directories if not there yet
    Utils.create_dirs_if_not_exist([FEATURES_DIR])


    @abc.abstractmethod
    def extract(self, dataset):
        pass

    @abc.abstractmethod
    def extract_from_timeseries(self, dataset):
        pass

    @abc.abstractmethod
    def save_features(self, dataset_name, features):
        pass

    @abc.abstractmethod
    def load_features(self, dataset):
        pass

    @abc.abstractmethod
    def _get_features_filename(self, dataset_name):
        pass

    def are_features_created(self, dataset_name):
        """
        Checks whether the features of the specified data set exist or not.
        
        Keyword arguments: 
        dataset_name -- name of the data set for which we check if the features exist
        
        Return: 
        True if the features have already been computed and saved as CSV, false otherwise.
        """
        features_filename = self._get_features_filename(dataset_name)
        return os.path.isfile(features_filename)