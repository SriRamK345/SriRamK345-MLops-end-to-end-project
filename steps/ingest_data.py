import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Initialize the IngestData class.

    """
    def __init__(self, data_path):

        """
        Arug:
            data_path (str): The path to the data
        """
        
        self.data_path = data_path

    def load_data(self):

        """
        Ingesting data from a data path
        """
        logging.info("Loading data from %s", self.data_path)
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a data path
    
    Args:
        data_path (str): The path to the data to be ingested
    
    Returns:
        pd.DataFrame: The ingested data
    """

    try:
        return IngestData(data_path).load_data()
    except Exception as e:
        logging.error("Error ingesting data: %s", e)
        raise e

