import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import *
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils import save_object

def main():
    try:
        obj=DataIngestion()
        train_data,test_data=obj.initiate_data_ingestion()
        print('Data ingestion finished!')
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
        print('Data transformation finished!')
        model_trainer = ModelTrainer()
        _ = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print('Model trainer finished!')
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()