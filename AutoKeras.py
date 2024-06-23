# coding: utf-8
import autokeras

class AutoKeras:
    '''
    Оболочка вокруг autoKeras для использования в Pipeline
    '''
    
    def fit(self, X, y):
        self.model.fit(X, y, epochs = 10, verbose = 0)
        
    def predict(self, X):
        return self.model.predict(X).flatten()
    
    def score(self, X, y):
        return self.model.evaluate(X, y)
    
    
class AutoKerasRegressor(AutoKeras):
    def __init__(self, **kwargs):
        self.model = autokeras.StructuredDataRegressor(**kwargs)
        
        
class AutoKerasClassifier(AutoKeras):
    def __init__(self, **kwargs):
        self.model = autokeras.StructuredDataClassifier(**kwargs)

