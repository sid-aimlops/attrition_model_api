import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from attrition_model.config.core import config
from attrition_model.pipeline import attrition_pipe
from attrition_model.processing.features import create_preprocessor
from attrition_model.processing.data_manager import load_dataset, save_pipeline

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier



def run_training() -> None:
    
    """
    Train the model.
    """    
    
    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    
    X =  data[config.model_config.features]
    y =  data[config.model_config.target]
    
    #print(f"X: {X}")
    #print(f"Y: {Y}")
        #    # Identify the numerical and categorical columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    #print(f"numerical_features {numerical_features}")
    #print(f"categorical_features {categorical_features}")
    
    # Create the preprocessing pipeline
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    #print(preprocessor)
    # Create the full pipeline with a classifier
    attrition_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=config.model_config.random_state))
    ])
    #print(attrition_pipeline)
   
        # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,  # predictors
        y,
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    attrition_pipeline.fit(X_train,y_train)
        # Predict and evaluate the model
    y_pred = attrition_pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
        # persist trained model
    save_pipeline(pipeline_to_persist= attrition_pipeline)


if __name__ == "__main__":
    run_training()