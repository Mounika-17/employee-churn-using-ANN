import sys
import pandas as pd
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            # Load preprocessor and model
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(model_path)

            # Apply transformations
            data_scaled = preprocessor.transform(features)

            # Predict churn probability
            preds = model.predict(data_scaled)
            churn_prob = preds[0][0]
            churn_label = 1 if churn_prob > 0.5 else 0

            return churn_label, churn_prob

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 CreditScore: float,
                 Geography: str,
                 Gender: str,
                 Age: float,
                 Tenure: int,
                 Balance: float,
                 NumOfProducts: int,
                 HasCrCard: int,
                 IsActiveMember: int,
                 EstimatedSalary: float):
        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.CreditScore],
                "Geography": [self.Geography],
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Tenure": [self.Tenure],
                "Balance": [self.Balance],
                "NumOfProducts": [self.NumOfProducts],
                "HasCrCard": [self.HasCrCard],
                "IsActiveMember": [self.IsActiveMember],
                "EstimatedSalary": [self.EstimatedSalary]
            }

            df = pd.DataFrame(custom_data_input_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)
