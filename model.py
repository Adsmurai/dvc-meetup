import random

class Model():
    def __init__(self):
        random.seed(42)

    def train(self, X, y):
        self._targets = y.unique().tolist()
        
    def predict(self, x):
        return [random.choice(self._targets) for _ in x]