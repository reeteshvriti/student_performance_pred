# here we will be performing feature engineering and data cleaning

import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.compose import ColumnTransformer #this helps us in creating the pipeline for onehot encoding and standardscaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    PrePocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_features = ['reading_score', 'writing_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                            'lunch', 'test_preparation_course']
            

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
            
            cat_pipeline = Pipeline(
                    steps= [
                     ("imputer", SimpleImputer(strategy="most_frequent")),   
                     ("one_hot_encoder", OneHotEncoder()) , 
                     ("scaler", StandardScaler(with_mean=False))
                    ]

            )

            logging.info(f"numerical columns scaling compleated :{num_features}")
            logging.info(f"categorical columns encoding compleated:{cat_features}")


            preprocessor = ColumnTransformer(
                [
                  ("num_pipeline", num_pipeline, num_features),
                  ("cat_pipeline", cat_pipeline, cat_features),    
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
            
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df =  pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(train_df.head())
            logging.info(test_df.head())

            logging.info("read train and test dataset")

            # logging.info("**********************")    

            logging.info("obtaining pre processing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_col = "math_score"
            num_col = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_col], axis=1)
            target_feature_train_df = train_df[target_col]
            
            input_feature_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("applying preprocessing object on training dataframe and testing dataframe")    

            

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # logging.info(train_arr, test_arr)    

            logging.info("saved preprocessing object")
            
            save_object(
                file_path =self.data_transformation_config.PrePocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.PrePocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e, sys)

    
