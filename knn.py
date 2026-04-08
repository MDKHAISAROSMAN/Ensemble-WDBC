import numpy as np
from collections import Counter
import pandas as pd

def euclidian_distance(a,b):
    return np.sqrt(np.sum((b-a)**2))

class KNN:
    def __init__(self,k):
        self.k  = k
    
    def train(self,x,y):
        self.x_train = x
        self.y_train = y

    def predict(self,new_input):
        prediction = [self.predict_class(new_point) for new_point in new_input]
        return prediction
    
    def predict_class(self,new_point):
        distances = [euclidian_distance(point,new_point) for point in self.x_train]
        k_nearest_index = np.argsort(distances)[:self.k]
        k_nearest_label = [self.y_train[i] for i in k_nearest_index]
        most_common = Counter(k_nearest_label).most_common(1)[0][0]
        return int(most_common)

def K_NN(name,x_train,x_test,y_train,y_test,test= None):
    name = KNN(7)
    x_test=x_test.astype(float)
    x_train = x_train
    name.train(x_train,y_train)
    predictions = name.predict(x_test)
    accuracy = np.mean(predictions == y_test) * 100
    if test.all() == None:
        return 0,accuracy
    else:
        test = np.array(test, dtype=float).reshape(1, -1)
        y = name.predict(test)
        return y,accuracy


