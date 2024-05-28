# coding: utf-8

import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from ML_model import Model

random_state_param = {'random_state' : 100}

class Classificator(Model):
    ''' Подбоор модели для задачи классификации
    '''
    
    def __init__(self):
        self.models = {'Logistic Regression': (LogisticRegression(), {}),
                       'SVM' : (SVC(**random_state_param), {'__gamma' : ['scale', 'auto']}),
                       'Random Forest' : (RandomForestClassifier(**random_state_param),
                                           {'__n_estimators': np.linspace(10, 1000, 7).round().astype('int'),
                                            '__max_features': np.linspace(0.1, 1, 5)}),
                       'XGBoost' : (XGBClassifier(**random_state_param),
                                     {'__n_estimators': np.linspace(10, 1000, 6).round().astype('int'),
                                      '__max_depth' : [2, 3, 4], 
                                      '__learning_rate' : [0.01, 0.05, 0.1, 0.2, 0.5, 0.8],
                                      '__subsample': [0.5, 0.7, 1.0]})
                      }
        
        self.metrics = ['f1']

        
    def Prepare_data(self):
        '''
        Подготовка данных для обучения моделей
        '''
        super().Prepare_data()
        
        # Кодирование целевой переменной от 0 до n_classes-1
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.y)
        
    
    def predict(self, X_test):
        if self.model:
            X_test = pd.DataFrame(X_test)[self.columns]
            return self.le.inverse_transform(self.model.predict(X_test))
        
        raise ValueError('Модель отсутствует.')
            
    
    def Remove_outliers(self, df):
        return df
