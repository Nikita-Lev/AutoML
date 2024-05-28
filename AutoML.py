#!/usr/bin/env python
# coding: utf-8

from Regression import Regressor
from Classification import Classificator

class autoML:
    ''' Автоматический подбор лучшей ML или DL модели для задачи регрессии, класификации, прогнозирования временных рядов
    '''
    
    def __new__(self, task):
        ''' 
        Parameters
        ----------
        task : {'Regression', 'Classification', 'Clusterization' 'Time Series'}
            Решаемая задача:
            Regression – предсказание вещественного значения целевой переменной
            Classification – предсказание вероятности принадлежности объектов к классу
            Clusterization – выделение кластеров объектов, схожих между собой
            Time Series – прогнозирование временного ряда по его значениям и иным признакам
        '''
        
        if task == 'Regression':
            return Regressor()
        if task == 'Classification':
            return Classificator()
        if task == 'Clusterization':
            return Clusterizator()
        if task == 'Time Series':
            return Time_Series()
        else:
            raise ValueError(f"Неизвестная задача: {task}")
           

