import random

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Model():
    def train(self, X, y):
        self._model = DecisionTreeClassifier(random_state=42)
        self._model.fit(X=X, y=y) # skipped parameter tunning for simplycity
        
    def predict(self, X):
        return self._model.predict(X)