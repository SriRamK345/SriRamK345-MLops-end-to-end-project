import logging
import pandas as pd
from zenml import step

@step
def evaluation_data(df: pd.DataFrame) -> None:
    pass