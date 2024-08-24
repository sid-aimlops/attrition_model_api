# tests/test_predict.py

import unittest
import pandas as pd
from attrition_model.predict import make_prediction

class TestMakePrediction(unittest.TestCase):
    
    def setUp(self):
        """Set up some sample input data for prediction."""
        self.sample_input_dict = {
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
            "trainingtimeslastyear": 3,
            "worklifebalance": 2,
            "yearsatcompany": 5,
            "yearsincurrentrole": 3,
            "yearssincelastpromotion": 2,
            "yearswithcurrmanager": 3
        }

        # Create a DataFrame version of the input
        self.sample_input_df = pd.DataFrame([self.sample_input_dict])

    def test_make_prediction_with_dict(self):
        """Test the make_prediction function with a dictionary input."""
        result = make_prediction(input_data=self.sample_input_dict)
        
        # Check the structure of the result
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('input_data', result)
        
        # Check that predictions is a list of the same length as input data
        self.assertIsInstance(result['predictions'], list)
        self.assertEqual(len(result['predictions']), 1)  # Single prediction

        # Check that input_data matches the provided input
        self.assertEqual(result['input_data'][0], self.sample_input_dict)

    def test_make_prediction_with_dataframe(self):
        """Test the make_prediction function with a DataFrame input."""
        result = make_prediction(input_data=self.sample_input_df)
        
        # Check the structure of the result
        self.assertIsInstance(result, dict)
        self.assertIn('predictions', result)
        self.assertIn('input_data', result)
        
        # Check that predictions is a list of the same length as input data
        self.assertIsInstance(result['predictions'], list)
        self.assertEqual(len(result['predictions']), 1)  # Single prediction

        # Check that input_data matches the provided input
        self.assertEqual(result['input_data'], self.sample_input_df.to_dict(orient='records'))

if __name__ == '__main__':
    unittest.main()