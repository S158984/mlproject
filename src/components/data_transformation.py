<<<<<<< HEAD
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exceptions import CustomException
from src.logger import logging
#from src.components.data_injestion import DataIngestion
from src.utils import save_object



@dataclass
class DataTransformationClass:
    preprocessor_object_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationClass()

    def get_data_transfer_object(self):
        try:
            num_features=['reading_score','writing_score']
            cat_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                            'test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ('scaler',StandardScaler())
                ]
           )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                    ]
           )
            
            logging.info('cat columns encoding is completed')
            preprocessing=ColumnTransformer(
                [
                    ("cat_pipeline",cat_pipeline,cat_columns),
                    ('num_pipeline',num_pipeline,num_features)
                ]
            )
            return preprocessing   
        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('reading of tarin and test data is done')

            preprocessing_obj=self.get_data_transfer_object()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            print('++++++++',input_feature_train_df.shape)

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path,
            )

        except Exception as e:
=======
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exceptions import CustomException
from src.logger import logging
#from src.components.data_injestion import DataIngestion
from src.utils import save_object



@dataclass
class DataTransformationClass:
    preprocessor_object_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationClass()

    def get_data_transfer_object(self):
        try:
            num_features=['reading_score','writing_score']
            cat_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch',
                            'test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean")),
                    ('scaler',StandardScaler())
                ]
           )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                    ]
           )
            
            logging.info('cat columns encoding is completed')
            preprocessing=ColumnTransformer(
                [
                    ("cat_pipeline",cat_pipeline,cat_columns),
                    ('num_pipeline',num_pipeline,num_features)
                ]
            )
            return preprocessing   
        except Exception as e:
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('reading of tarin and test data is done')

            preprocessing_obj=self.get_data_transfer_object()
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df=train_df[target_column_name]
            print('++++++++',input_feature_train_df.shape)

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path,
            )

        except Exception as e:
>>>>>>> 1147cfc65f5db45000d3ad34f292c2025e43b176
            raise CustomException(e,sys)