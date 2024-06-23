# coding: utf-8

import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from AutoKeras import AutoKerasRegressor

from ML_model import Model

random_state_param = {'random_state' : 100}

class Regressor(Model):
    ''' Подбоор модели для задачи регрессии
    '''
    def __init__(self):
        self.models = {
                        'Linear Regression': (LinearRegression(), {}),
                        'Lasso Regression': (Lasso(**random_state_param), {'__alpha' : [0.1, 0.5, 1]}),
                        'Ridge Regression': (Ridge(**random_state_param), {'__alpha' : [0.1, 0.5, 1]}),
                        'SVM' : (SVR(), {'__gamma' : ['scale', 'auto']}),
                        'Random Forest' : (RandomForestRegressor(**random_state_param),
                                           {'__n_estimators': np.linspace(10, 1000, 7).round().astype('int'),
                                            '__max_features': np.linspace(0.1, 1, 5)}),
                        'CatBoost' : (CatBoostRegressor(**random_state_param),
                                     {'__iterations': [100, 500, 1000],
                                      '__depth' : [2, 3, 4], 
                                      '__learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.5, 0.8],
                                      '__l2_leaf_reg': [1, 3, 5, 7],
                                      '__bagging_temperature': [0.0, 0.5, 1.0]}),
                        'XGBoost' : (XGBRegressor(**random_state_param),
                                     {'__n_estimators': np.linspace(10, 1000, 6).round().astype('int'),
                                      '__max_depth' : [2, 3, 4], 
                                      '__learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.5, 0.8],
                                      '__subsample': [0.5, 0.7, 1.0]}),
                        'Auto Keras' : (AutoKerasRegressor(max_trials = 10, overwrite = True), {})
                       
                      }
        
        self.metrics = ['neg_mean_squared_error']

