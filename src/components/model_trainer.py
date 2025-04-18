import os
import sys
from dataclasses import dataclass
import pandas as pd


from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, 
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model


# creating a config file to save the model

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.Model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("entered into model trainer and splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:, -1], 
                test_array[:,:-1], test_array[:, -1]
            )


            models = {
            "Random_Forest":RandomForestRegressor(), 
            "Decesion_tree": DecisionTreeRegressor(),
            "knn": KNeighborsRegressor(),
            "Gradient_descent": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "CatBoost": CatBoostRegressor(verbose=False),
            "XGboost": XGBRegressor(),
            "LinearRegression": LinearRegression()    
             }
            
            model_report:dict = evaluate_model(x_train=X_train, y_train=y_train,
                                               x_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException("No best model found") 
            

            logging.info(f"best model found and is {best_model} with {round(best_model_score,2)} score ")

            save_object(
                file_path=self.Model_trainer_config.trained_model_path,
                obj = best_model
            ) 


            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted) 

            return r2_square

        except Exception as e:
            raise CustomException(e, sys) 
