import os
import sys
from src.exceptions import CustomException
from src.logger import logging
#from src.components.data_transformation import DataTransformationClass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngesionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')
    print('+++++++++++',train_data_path)

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngesionConfig()
        print('------>')##,self.ingestion_config.train_data_path)

    def initiate_data_ingestion(self):
        logging.info('insdie data reading')
        logging.info('------>>>>>>>  insdie data reading')
        try:
            df1=pd.read_csv("src\\notebook\\data\\stud.csv")
            logging.info('data read successfully') 
            #print('--->',os.path.dirname()) 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            #os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            #os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            #os.makedirs('TestDir',exist_ok=True)

            df1.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('data saved successfully')

            train_set, test_set=train_test_split(df1, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("data save to folders succesfully.")
            return (
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path,

            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_train(train_arr,test_arr))

   