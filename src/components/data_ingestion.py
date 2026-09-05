import os
import sys
from src.exception import CustomerException
from src.logger import logging
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataIngestion:
    def __init__(self, input_path, output_path):
          self.input_path = input_path  
          self.output_path = output_path
    def read_data(self):
        try:
            print('Start reading the data')
            logging.info("Enter the read method to read the dataset")
            
            df =   pd.read_csv(self.input_path, header=None, names=["id", "game", "sentiment", "text"], sep=",")
            
            logging.info("Done reading the dataset")
            print('Done reading the data')
            
            return df
        except Exception as e:
            raise CustomerException(e, sys)
            print("The dataset is not present.")
    
    def save_raw_data(self, data):
        try:
            logging.info("Enter the save raw data method to save the dataset")
            print("Enter the save raw data method to save the dataset")
            
            os.makedirs(self.output_path, exist_ok=True)
            raw_data_file_path = os.path.join(self.output_path, 'raw_data.csv')
            data.to_csv(raw_data_file_path, index=False)
            
            print("Done save  method to save the dataset")
            logging.info("Done save  method to save the dataset")
        except Exception as e:
            raise CustomerException(e, sys)
            print("Error comes in saving dataset")
        
    def save_train_test_data(self, data):
        try:
            print("\nSplitting the dataset")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            print("\n Done splitting the dataset")
            
            print("Now saving the raw training and testing dataset")
            os.makedirs(self.output_path, exist_ok=True)
            raw_train_file_path=os.path.join(self.output_path,'raw_train.csv')
            train_data.to_csv(raw_train_file_path, index=False)
            print("Done saving the raw train dataset")
            
            raw_test_file_path=os.path.join(self.output_path, "raw_test.csv")
            test_data.to_csv(raw_test_file_path)
            print("Done saving the raw test dataset")
        except Exception as e:
            raise CustomerException(e, sys) 

# if __name__=="__main__":    
#     di = DataIngestion(input_path='test_data/twitter_dataset.csv',output_path='Data/raw')
#     data=di.read_data()
#     di.save_raw_data(data)
#     di.save_train_test_data(data)
