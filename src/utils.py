import os
import sys
import numpy as np
import pandas as pd
import dill  
import pickle
from src.exceptions import CustomException
from src.logger import logging

from sklearn.metrics import r2_score    #importing r2_score from sklearn.metrics




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        logging.info('saev object is called.')
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train,y_train, x_test, y_test,models):
    try:
        report = {}
        print(len(models))
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(x_train, y_train)
            y_train_pred=model.predict(x_train) 
            y_test_pred=model.predict(x_test)
            train_model_score=r2_score(y_train, y_train_pred) 
            test_model_score=r2_score(y_test, y_test_pred)
            print('train_model_score:',train_model_score)
            logging.info(f"train_model_score: {train_model_score}")
            report[list(models.keys())][i]=test_model_score
        return report

    except:
        pass