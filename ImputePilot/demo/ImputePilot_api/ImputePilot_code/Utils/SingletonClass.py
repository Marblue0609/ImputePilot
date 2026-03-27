"""
ImputePilot: A Demo System for Stable Imputation Model Selection in Time Series Data Repair
Zhejiang University
***
SingletonClass.py
@author: zhexinjin@zju.edu.cn
"""

import abc

class SingletonClass(metaclass=abc.ABCMeta):
    """
    Singleton class.
    """

    _INSTANCE = None
    
    
    # static methods

    @abc.abstractmethod
    def get_instance():
        pass

    def get_instance(cls):
        """
        Returns the single instance of this class.
        
        Keyword arguments: 
        cls -- class inheriting from SingletonClass from which an instanced is requested.
        
        Return: 
        Single instance of this class.
        """
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(caller='get_instance')
        return cls._INSTANCE