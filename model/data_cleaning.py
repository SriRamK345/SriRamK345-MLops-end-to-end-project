import logging
import pandas as pd
from zenml import step
from sklearn.impute import SimpleImputer

from abc import ABC, abstractmethod

class DataStrategy(ABC):
    """
    
    """