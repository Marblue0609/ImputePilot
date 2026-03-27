"""
ImputePilot: A Demo System for Stable Imputation Model Selection in Time Series Data Repair
Zhejiang University
***
Dataset.py
@author: zhexinjin@zju.edu.cn
"""

import os
from os.path import normpath as normp
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import re
from tqdm import tqdm
import yaml
import zipfile

from ImputePilot_api.ImputePilot_code.Utils.Utils import Utils

'''
class Dataset:
    """
    Class which handles a time series data set.
    """

    RW_DS_PATH = normp('./Datasets/RealWorld/')
    CONF = Utils.read_conf_file('datasets')
'''

class Dataset:
    """
    Class which handles a time series data set.
    """
    
    # 获取当前 Dataset.py 文件所在的目录，并拼接出 RealWorld 文件夹的绝对路径
    RW_DS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RealWorld')
    
    # 确保文件夹存在，防止报错
    if not os.path.exists(RW_DS_PATH):
        os.makedirs(RW_DS_PATH)

    CONF = Utils.read_conf_file('datasets')


    # constructor

    def __init__(self, archive_name, clusterer, data_dir=None):
        self.rw_ds_filename = archive_name
        self.rw_ds_dir = data_dir if data_dir is not None else Dataset.RW_DS_PATH
        self.name = os.path.splitext(archive_name)[0]
        self.nb_timeseries, self.timeseries_length = self.load_timeseries(transpose=True).shape

        self.clusterer = clusterer
        self.cids = None
        if self.clusterer.are_clusters_created(self.name):
            cassignment = self.load_cassignment(self.clusterer)
            self.cids = cassignment['Cluster ID'].unique().tolist()


    # public methods

    def get_cluster_by_id(self, timeseries, cluster_id, cassignment):
        """
        Returns the time series belonging to the specified cluster's id.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series).
        cluster_id -- cluster id (int) of which the time series that must be returned belong to.
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series.
        
        Return:
        Pandas DataFrame containing the time series belonging to the specified cluster's id (each row is a time series).
        """
        return timeseries.loc[cassignment['Time Series ID'][cassignment['Cluster ID'] == cluster_id]]

    def yield_all_clusters(self, timeseries, cassignment=None):
        """
        Yields the time series of each cluster in a Pandas DataFrame.
        
        Keyword arguments:
        timeseries -- Pandas DataFrame containing the time series (each row is a time series).
        cluster_id -- cluster id (int) of which the time series that must be returned belong to.
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series (default None, if None, loads it).
        
        Return:
        1. Pandas DataFrame containing the time series belonging to the specified cluster's id (each row is a time series).
        2. Cluster id (int) of the cluster being returned.
        3. Pandas DataFrame containing clusters' assignment of the data set's time series. Its index is the same as the real 
           world data set of this object. The associated values are the clusters' id to which are assigned the time series.
        """
        # load clusters assignment
        if cassignment is None:
            cassignment = self.load_cassignment(self.clusterer)
        for cluster_id in cassignment['Cluster ID'].unique(): # for each cluster ID present in this dataset
            # retrieve time series assigned to this cluster
            cluster = self.get_cluster_by_id(timeseries, cluster_id, cassignment)
            yield cluster, cluster_id, cassignment

    def load_timeseries(self, transpose=False):
        """
        Loads time series which are stored in a .txt file with either a .info or a .index file describing the dates used as index.
        
        Keyword arguments:
        transpose -- transpose the data set if true (default False)
        
        Return:
        Pandas DataFrame containing the time series
        """
        ds_filename_ext = ('txt', 'csv', 'tsv')
        index_filename_ext = 'index'
        info_filename_ext = 'info' 

        def _get_filename(candidates, ext):
            try:
                return next(x for x in candidates if x.lower().endswith(ext))
            except StopIteration:
                return None

        def _apply_datetime_index(dataset, index_loader=None, info_loader=None):
            if self._is_datetime_col(dataset[0]): # if first column is of type datetime: use it as index
                dataset['DateTime'] = pd.to_datetime(dataset[0])
                dataset = dataset.drop(columns=[0])
                dataset = dataset.set_index('DateTime')
                dataset.columns = pd.RangeIndex(dataset.columns.size)
            else: # else try to load an index or info file containing information about the data set's index
                if index_loader is not None:
                    # index: each point's date is specified in file
                    index = index_loader()
                    dataset['DateTime'] = index[0].tolist()
                    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'])
                    dataset = dataset.set_index('DateTime')
                elif info_loader is not None:
                    # info: only start date, periods and freq are given, date range is created from this
                    date_range = info_loader()
                    date_range = dict(zip(date_range.columns, date_range.values[0]))
                    dataset = dataset.set_index(pd.date_range(**date_range))
                else:
                    date_range = {'start': '1900-1-1', 'periods': len(dataset.index), 'freq': '1s'}
                    dataset = dataset.set_index(pd.date_range(**date_range))
            return dataset

        file_path = normp(os.path.join(self.rw_ds_dir, self.rw_ds_filename))
        ext = os.path.splitext(self.rw_ds_filename)[1].lower()

        if ext == '.zip':
            with zipfile.ZipFile(file_path, 'r') as archive:
                all_names = archive.namelist()
                
                # Try matching by dataset name first
                r = re.compile(f'.*{re.escape(self.name)}.*\\.(?:txt|csv|tsv|index|info)$', re.IGNORECASE)
                filenames = list(filter(r.match, all_names))
                
                # Fallback: if no match by name, search all files by extension
                if not filenames:
                    filenames = [f for f in all_names if f.lower().endswith(('.txt', '.csv', '.tsv', '.index', '.info'))]
                    if filenames:
                        print(f"[WARN] Dataset '{self.name}': name-based match failed, using extension-based fallback: {filenames}")
                    else:
                        raise FileNotFoundError(
                            f"No data file (.txt/.csv/.tsv) found in '{self.rw_ds_filename}'. "
                            f"Zip contents: {all_names}"
                        )

                ds_filename = _get_filename(filenames, ds_filename_ext)
                if ds_filename is None:
                    raise FileNotFoundError(f"No dataset matrix file found in archive: {self.rw_ds_filename}")
                dataset = pd.read_csv(archive.open(ds_filename), sep=None, engine='python', header=None)

                index_filename = _get_filename(filenames, index_filename_ext)
                info_filename = _get_filename(filenames, info_filename_ext)
                dataset = _apply_datetime_index(
                    dataset,
                    index_loader=(lambda: pd.read_csv(archive.open(index_filename), sep=' ', header=None, parse_dates=True))
                    if index_filename is not None else None,
                    info_loader=(lambda: pd.read_csv(archive.open(info_filename), sep=' ', parse_dates=True))
                    if info_filename is not None else None,
                )

        elif ext in ('.txt', '.csv', '.tsv'):
            dataset = pd.read_csv(file_path, sep=None, engine='python', header=None)
            index_path = os.path.join(self.rw_ds_dir, f'{self.name}.index')
            info_path = os.path.join(self.rw_ds_dir, f'{self.name}.info')
            dataset = _apply_datetime_index(
                dataset,
                index_loader=(lambda: pd.read_csv(index_path, sep=' ', header=None, parse_dates=True))
                if os.path.exists(index_path) else None,
                info_loader=(lambda: pd.read_csv(info_path, sep=' ', parse_dates=True))
                if os.path.exists(info_path) else None,
            )
        else:
            raise ValueError(f"Unsupported dataset file extension: {self.rw_ds_filename}")

        return dataset if not transpose else dataset.T

    def save_cassignment(self, clusterer, cassignment):
        """
        Saves the given clusters to CSV.
        
        Keyword arguments: 
        clusterer -- clusterer instance used to create the clusters to load
        cassignment -- Pandas DataFrame containing clusters' assignment of the data set's time series. 
                       Its index is the same as the real world data set of this object. The associated 
                       values are the clusters' id to which are assigned the time series.
        
        Return: -
        """
        clusterer.save_clusters(self, cassignment)

    def load_cassignment(self, clusterer):
        """
        Loads the Pandas Dataframe containing the clusters' assignment of this data set. 
        
        Keyword arguments: 
        clusterer --- clusterer instance used to create the clusters to load
        
        Return:
        Pandas DataFrame containing clusters' assignment of the data set's time series. Its index is the same 
        as the real world data set of this object. The associated values are the clusters' id to which are
        assigned the time series. Two columns: Time Series ID, Cluster ID.
        """
        return clusterer.load_clusters(self.name)

    def get_space_complexity(self):
        """
        Computes and returns the space complexity of the data set's time series.
        
        Keyword arguments: -
        
        Return:
        Space complexity of the data set's time series
        """
        return self.nb_timeseries * self.timeseries_length

    def load_labels(self, labeler, properties):
        """
        Loads the labels created using the given labeler and defined by the specified properties.
        
        Keyword arguments: 
        labeler -- labeler instance used to create the labels to load
        properties -- dict specifying the labels' properties
        
        Return: 
        1. Pandas DataFrame containing the data set's labels. Two columns: Time Series ID and Label.
        2. List of all possible labels value
        """
        return labeler.load_labels(self, properties)

    def save_labels(self, labeler, labels):
        """
        Saves the given labels to CSV.
        
        Keyword arguments: 
        labeler -- labeler instance used to create the labels to save
        labels -- Pandas DataFrame containing the labels to save. 
        
        Return: -
        """
        labeler.save_labels(self.name, labels)

    def load_features(self, features_extractor):
        """
        Loads the features created using the given extractor.
        
        Keyword arguments: 
        features_extractor -- features extractor instance used to create the features to load
        
        Return: 
        Pandas DataFrame containing the data set's features. Each row is a time series feature vector.
        Columns: Time Series ID, (Cluster ID), Feature 1's name, Feature 2's name, ...
        """
        return features_extractor.load_features(self)

    def save_features(self, features_extractor, features):
        """
        Saves the given features to CSV.
        
        Keyword arguments: 
        features_extractor -- features extractor instance used to create the features to save
        features -- Pandas DataFrame containing the features to save. 
        
        Return: -
        """
        features_extractor.save_features(self.name, features)


    # private methods

    def __repr__(self):
        return self.name
    
    def _is_datetime_col(self, col):
        """
        Checks if a Pandas Series is of type date time.
        
        Keyword arguments:
        col -- Pandas series
        
        Return:
        True if the Pandas Series contains only date time objects False otherwise
        """
        if col.dtype == 'object':
            try:
                col = pd.to_datetime(col)
                return True
            except ValueError:
                return False
        return is_datetime(col)


    # static methods

    @staticmethod
    def instantiate_from_dir(clusterer, data_dir=None):
        """
        Instantiates multiple data set objects from the Datasets/RealWorld folder.
        Uses the Datasets conf file to define which data set use.
        
        Keyword arguments: 
        clusterer --- clusterer instance that will be (or has been) used to cluster this data set's time series
        
        Return:
        List of Dataset objects
        """
        root_dir = data_dir if data_dir is not None else Dataset.RW_DS_PATH
        if Dataset.CONF['USE_ALL']: 
            # or ds_filename in Dataset.CONF['USE_LIST']]
            valid_extensions = ('.zip', '.csv', '.txt', '.tsv')
            timeseries = [
                Dataset(ds_filename, clusterer, data_dir=root_dir) 
                for ds_filename in os.listdir(root_dir) 
                if ds_filename.lower().endswith(valid_extensions)
                ]
        else:
            timeseries = []
            for ds_filename in Dataset.CONF['USE_LIST']:
                if os.path.exists(os.path.join(root_dir, ds_filename)):
                    timeseries.append(Dataset(ds_filename, clusterer, data_dir=root_dir))
        
        # check: either use all data sets listed in the folder
        # ... or verify that all data sets listed in the conf file have been found and loaded
        assert Dataset.CONF['USE_ALL'] or len(timeseries) == len(Dataset.CONF['USE_LIST'])

        return timeseries

    @staticmethod
    def yield_each_datasets_cluster(datasets):
        """
        One-by-one, yields each cluster of all given datasets.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series and their clusters assignment.
        
        Return: 
        1. Dataset object which contains the yielded cluster
        2. Pandas DataFrame containing all the time series of the data set (each row is a time series)
        3. Pandas DataFrame containing the time series belonging to one of the clusters to yield (each row is a time series).
        4. ID of the yielded cluster
        """
        for dataset in tqdm(datasets): # for each data set
            timeseries = dataset.load_timeseries(transpose=True) # load data set's time series
            clusters_assignment = dataset.load_cassignment(dataset.clusterer) # load clusters assignment
            for cluster_id in clusters_assignment['Cluster ID'].unique(): # for each cluster
                cluster = dataset.get_cluster_by_id(timeseries, cluster_id, clusters_assignment)
                yield dataset, timeseries, cluster, cluster_id

    @staticmethod
    def yield_each_datasets_cluster_id(datasets):
        """
        One-by-one, yields each cluster id of all given datasets.
        
        Keyword arguments:
        datasets -- list of Dataset objects containing the time series and their clusters assignment.
        
        Return: 
        ID of the yielded cluster
        """
        for dataset in datasets: # for each data set
            for cid in dataset.cids:
                yield cid
