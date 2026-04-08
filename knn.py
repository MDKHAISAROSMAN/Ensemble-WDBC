import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path,sep=",",header=None,na_values=[0])
    df=df.dropna(axis = 0)
    label = (df[1] == "M").astype(int)
    df=df.drop([0,1],axis=1)
    df.columns =[f"feature{i}" for i in range(1,df.shape[1]+1)]
    label.columns = ["result"]
    return df,label

def split_data(df,label):
    x_train,x_test,y_train,y_test = train_test_split(df,label,random_state=42,stratify=label,test_size=0.25)
    return x_train,x_test,y_train,y_test

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

def K_NN(name,path,test= None):
    df,label = load_data(path)
    x_train,x_test,y_train,y_test = split_data(df,label)
    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.values
    y_test = y_test.values
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


