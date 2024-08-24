import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union, Dict
import pandas as pd
import numpy as np

from attrition_model import __version__ as _version
from attrition_model.config.core import config
from attrition_model.processing.data_manager import load_pipeline

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
attrition_pipe= load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> Dict:
    """
    Make a prediction using the trained model pipeline.

    Parameters:
    - input_data (Union[pd.DataFrame, dict]): The input data to make predictions on.
      Can be a DataFrame or a dictionary.

    Returns:
    - Dict: A dictionary containing the predictions and the model's input data.
    """
    # Ensure the input data is a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])  # Convert dict to DataFrame
        #print(f"input_data: {input_data}")
        
        
    #input_data=input_data.reindex(columns=config.model_config.features)
    #print(validated_data)
    #results = {"predictions": None, "version": _version}
    

    
    # Make predictions
    predictions = attrition_pipe.predict(input_data)
    
    # Convert the predictions to a human-readable format
    predicted_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
    
    # Return the predictions in a dictionary
    return {
        'predictions': predicted_labels,
        'input_data': input_data.to_dict(orient='records')
    }
if __name__ == "__main__":
    # Example usage:
    sample_input = {
        "age": 34,
        "businesstravel": "Travel_Rarely",
        "dailyrate": 1000,
        "department": "Sales",
        "distancefromhome": 10,
        "education": 3,
        "educationfield": "Life Sciences",
        "employeecount": 1,
        "employeenumber": 1,
        "environmentsatisfaction": 3,
        "gender": "Male",
        "hourlyrate": 80,
        "jobinvolvement": 3,
        "joblevel": 2,
        "jobrole": "Sales Executive",
        "jobsatisfaction": 4,
        "maritalstatus": "Married",
        "monthlyincome": 5000,
        "monthlyrate": 15000,
        "numcompaniesworked": 2,
        "over18": "Y",
        "overtime": "Yes",
        "percentsalaryhike": 12,
        "performancerating": 3,
        "relationshipsatisfaction": 3,
        "standardhours": 80,
        "stockoptionlevel": 1,
        "totalworkingyears": 10,
        "trainingtimesLastyear": 3,
        "worklifebalance": 2,
        "yearsatcompany": 5,
        "yearsincurrentrole": 3,
        "yearssinceLlastpromotion": 2,
        "yearswithcurrmanager": 3
    }
    
    prediction_result = make_prediction(input_data=sample_input)
    print(prediction_result)