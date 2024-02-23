import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model
from sklearn.ensemble import RandomForestRegressor
import pkg_resources


def init():
    pass

def run(raw_data):
    try:
        # Convert the raw data (JSON) into a pandas DataFrame
        data = json.loads(raw_data)['data']
        new_inputs = pd.DataFrame(data, columns=['Candidate Skills', 'Candidate Experience', 'Candidate Education'])
        
        # Create a RandomForestRegressor model
        model = RandomForestRegressor()
        
        # Load the trained model from the Azure ML workspace
        # model_path = Model.get_model_path('model')  # 'model' is the name of the registered model in Azure ML
        # model = joblib.load(model_path)
        
        # Use the trained model to predict with new inputs
        new_predictions = model.predict(new_inputs)
        
        # Prepare the predictions as a JSON response
        result = {"predictions": new_predictions.tolist()}
        return result
    except Exception as e:
        error = str(e)
        return error
