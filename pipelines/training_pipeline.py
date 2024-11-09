from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation_data import evaluation_data

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path=data_path)
    clean_data(df)
    model_train(df)
    evaluation_data(df)  
    return