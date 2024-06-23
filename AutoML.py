#!/usr/bin/env python
# coding: utf-8

from Regression import Regressor
from Classification import Classifier
from Clusterization import Clusterizator

class autoML:
    ''' Автоматический подбор лучшей ML или DL модели для задачи регрессии, класификации, кластеризации
    '''
    
    def __new__(self, task):
        ''' 
        Parameters
        ----------
        task : {'Regression', 'Classification', 'Clusterization'}
            Решаемая задача:
            Regression – предсказание вещественного значения целевой переменной
            Classification – предсказание вероятности принадлежности объектов к классу
            Clusterization – выделение кластеров объектов, схожих между собой
        '''
        
        if task == 'Regression':
            return Regressor()
        if task == 'Classification':
            return Classifier()
        if task == 'Clusterization':
            return Clusterizator()
        else:
            raise ValueError(f"Неизвестная задача: {task}")
           

