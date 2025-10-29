import sys
import os
from dataclasses import dataclass
from src.logger import logger
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object
from src.config import categorical_encode,continuous_numerical_columns,target_column


from src.exception import CustomException

import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Custom Transformers
# -----------------------------
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import yeojohnson
# Handle the outliers
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson


# Outlier handler
class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, discrete_threshold=10, skew_threshold=0.75):
       # self.outlier_columns = outlier_columns
        self.discrete_threshold = discrete_threshold
        self.skew_threshold = skew_threshold
        self.params_ = {}
        self.outlier_columns = []
    # we need to give y=None as the scikit-learn’s Pipeline, GridSearchCV, cross_val_score, and other utilities always pass both X and y to .fit(), even for unsupervised transformers (like imputers, scalers, outlier handlers, etc.) even if they are not used.
    # This is to ensure compatibility with scikit-learn's API.
    def fit(self, X, y=None):
        X_ = X.copy()
        numeric_columns= X_.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_[col].dropna().empty:
                continue
            unique_values = X_[col].nunique()
            col_type = 'discrete' if unique_values <= self.discrete_threshold else 'continuous' 
            Q1 = X_[col].quantile(0.25)
            Q3 = X_[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            extreme_outliers = ((X_[col] < lower_bound) | (X_[col] > upper_bound)).sum()
            if extreme_outliers > 0:
                self.outlier_columns.append(col)
                if col_type == 'discrete':
                    self.params_[col] = ('cap', lower_bound, upper_bound)
                else:
                    skewness = X_[col].skew()
                    if abs(skewness) > self.skew_threshold:
                        if (X_[col] >= 0).all():
                            self.params_[col] = ('log',)
                        else:
                            self.params_[col] = ('yeojohnson',)
                    else:
                        self.params_[col] = ('cap', lower_bound, upper_bound)
        return self  # fit method returns the self which is a common practice in scikit-learn to allow for method chaining
    
    def transform(self, X):
        X_ = X.copy()
        for col, params in self.params_.items():
            if params[0] == 'cap':
                _, lower, upper = params
                #X_[col] < lower compares each value in the column to the lower bound. It returns a boolean Series where each entry is True if the corresponding value is less than the lower bound and False otherwise.
                X_[col] = np.where(X_[col] < lower, lower,
                                   np.where(X_[col] > upper, upper, X_[col]))
            elif params[0] == 'log':
                X_[col] = np.log1p(X_[col])
            elif params[0] == 'yeojohnson':
                X_[col], _ = yeojohnson(X_[col])
        return X_ # transform method returns the transformed DataFrame, allowing for further processing or model fitting

# -----------------------------
# Config
# -----------------------------

@dataclass
class DataTransformationConfig:
    # This defines where the preprocessing object (e.g., a fitted ColumnTransformer or Pipeline) will be saved after training.
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def build_preprocessor(self,continuous_numerical_columns, categorical_encode):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(),continuous_numerical_columns),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_encode),
            ],
            remainder='drop'
        )
        return preprocessor

    def get_data_transformer_object(self,continuous_numerical_columns, categorical_encode):
        '''Return a preprocessing pipeline (not fitted yet).'''
        try:
            preprocessor = Pipeline([
                ('outliers', OutlierHandler()),
                ('encode_scale',self.build_preprocessor(continuous_numerical_columns, categorical_encode) )
            ])
            logger.info("Preprocessing object created")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    # Now do the train test split and apply the transformation because we won't be performing the cross validation or GridSearchCV step. In ANN, the CV will be internally handled.So we are doing the train test split here and the transformation on the train and test data.
    def initiate_data_transformation(self, train_path, test_path, continuous_numerical_columns, categorical_encode, target_column):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Read train and test data completed")

            preprocessing_obj = self.get_data_transformer_object(
                continuous_numerical_columns=continuous_numerical_columns,
                categorical_encode=categorical_encode
            )

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logger.info("Applying preprocessing on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

        