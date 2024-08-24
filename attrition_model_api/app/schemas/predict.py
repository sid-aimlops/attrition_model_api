from typing import Any, List, Optional

from pydantic import BaseModel
from attrition_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
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
                ]
            }
        }
