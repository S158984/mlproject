import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from src.utils import evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor  

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exceptions import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    model_train_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_cofig=ModelTrainerConfig()

    def initiate_model_train(self, train_arr, test_arr):
        logging.info("Inside model training")
        x_train, y_train, x_test, y_test=(train_arr[:,:-1],train_arr[:,-1:],test_arr[:,:-1],test_arr[:,-1:])
        try:
            models={'randomForest':RandomForestRegressor(), 
                    'DecisionTree':DecisionTreeRegressor,
                    'GradientBoost':GradientBoostingRegressor(),
                    'LinearRegres':LinearRegression(),
                    'knn':KNeighborsRegressor()
                    #'XGBClassifier':XGBRegressor()
                    #'Catboost':CatBoostRegressor(),
                    #'Adaboost':AdaBoostRegressor(),
                    
                    }
            
            models_report: dict=evaluate_model(x_train=x_train, y_train=y_train,x_test=x_test, 
                                               y_test=y_test,models=models)
            logging.info('models_report  --------->>>>>> is ',models_report)
            best_model_score=max(sorted(models_report.values()))
            logging.info('best_model_score  ======> is ',best_model_score)
            best_model_name=list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
                                        ]
            if best_model_score<0.6:
                raise CustomException("no best Model availbale")
            logging.info('model training is completed.')                                  

            save_object(
                file_path=self.model_train_cofig.model_train_file_path,
                obj=best_model_name
                      ) 
            prediected=best_model_name(x_test)
            r2score=r2score(x_test, prediected)
            return r2score
        except  Exception as e:
            raise CustomException(e,sys)