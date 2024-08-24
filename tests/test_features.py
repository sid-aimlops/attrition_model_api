"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
import unittest
from sklearn.pipeline import Pipeline
from attrition_model.config.core import config
from attrition_model.processing.features import numerical_transformer, categorical_transformer, create_preprocessor
from sklearn.compose import ColumnTransformer

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        """Set up some sample data and column names."""
        self.numerical_features = ['Age', 'DailyRate', 'DistanceFromHome']
        self.categorical_features = ['BusinessTravel', 'Department', 'EducationField']

        self.sample_data = pd.DataFrame({
            'Age': [29, 35, 42, 50],
            'DailyRate': [500, 700, 1000, 1200],
            'DistanceFromHome': [10, 20, 5, 3],
            'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely'],
            'Department': ['Sales', 'HR', 'Development', 'Sales'],
            'EducationField': ['Life Sciences', 'Medical', 'Other', 'Medical']
        })

    def test_numerical_transformer(self):
        """Test if numerical transformer pipeline is created successfully and processes correctly."""
        transformer = numerical_transformer(self.numerical_features)
        self.assertIsInstance(transformer, Pipeline)
        
        transformed_data = transformer.fit_transform(self.sample_data[self.numerical_features])
        self.assertEqual(transformed_data.shape[1], len(self.numerical_features))  # Same number of features after transformation

    def test_categorical_transformer(self):
        """Test if categorical transformer pipeline is created successfully and processes correctly."""
        transformer = categorical_transformer(self.categorical_features)
        self.assertIsInstance(transformer, Pipeline)
        
        transformed_data = transformer.fit_transform(self.sample_data[self.categorical_features])
        self.assertGreater(transformed_data.shape[1], len(self.categorical_features))  # One-hot encoding increases number of features

    def test_create_preprocessor(self):
        """Test if the full preprocessor combines both numerical and categorical transformers."""
        preprocessor = create_preprocessor(self.numerical_features, self.categorical_features)
        self.assertIsInstance(preprocessor, ColumnTransformer)
        
        transformed_data = preprocessor.fit_transform(self.sample_data)
        expected_columns = len(self.numerical_features) + len(self.categorical_features)  # Base number before one-hot encoding
        self.assertGreaterEqual(transformed_data.shape[1], expected_columns)

if __name__ == '__main__':
    unittest.main()