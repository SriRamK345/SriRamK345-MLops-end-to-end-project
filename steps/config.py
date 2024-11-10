from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model configuration"""
    model_name: str = "LinearRegressionModel"
