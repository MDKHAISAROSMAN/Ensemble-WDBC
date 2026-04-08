import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def normalize(x_train, x_test,a):
    mean = np.mean(x_train, axis=0)
    mean = mean.values.reshape(1,-1)
    std = np.std(x_train, axis=0)
    std = std.values.reshape(1,-1)
    if a.all()==None:
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std
        return x_train,x_test,None
    else:
        x_train = (x_train - mean) / std
        x_test  = (x_test  - mean) / std
        a = (a - mean) / std
        return x_train, x_test,a
    
def split_data(df,label):
    x_train,x_test,y_train,y_test = train_test_split(df,label,random_state=42,stratify=label,test_size=0.25)
    return x_train,x_test,y_train,y_test

def normalize_new(new,mean,std):
    new = (new - mean) / std
    return new

class fit():
    def __init__(self,x_train,y_train,epoch,alpha):
         self.x_train = x_train
         self.y_train = y_train
         self.epoch = epoch
         self.alpha = alpha
         self.w,self.b = self.train(self.x_train,y_train,self.epoch,self.alpha)

    def train(self,x_train,y_train,epoch,alpha):
        m = y_train.shape[0]
        b = np.random.uniform(0,1,(1,1))
        w = np.random.uniform(0,1,(x_train.shape[1],1))
        for i in range(epoch):
            z = np.dot(x_train,w) + b
            z = np.clip(z,-500,500)
            A = 1/(1+np.exp(-z))
            dz = A - y_train
            dw = (x_train.T @ dz) / m
            db = np.sum(dz)/m
            w-=alpha*dw
            b-=alpha*db
            l = -np.mean(y_train * np.log(A + 1e-9) + (1 - y_train) * np.log(1 - A + 1e-9))
        #print(f"final loss: {l}")
        return w,b
    
    def predict(self,test_input):
        z = (test_input @ self.w)+self.b
        K = 1/(1+np.exp(-z))
        y_pred = (K>=0.5).astype(int)
        return y_pred


def regression(name,df,label,new= None):
    epoch = 200
    alpha = 0.5
    new = new.reshape(1,-1)
    x_train,x_test,y_train,y_test = split_data(df,label)
    x_train, x_test, new = normalize(x_train, x_test,new)
    name = fit(x_train, y_train, epoch, alpha)
    y_pred = name.predict(x_test)
    acc = np.mean(y_pred == y_test) * 100

    if(new.all() == None):
        return acc
    else:
        prediction = name.predict(new.reshape(1,-1))
        return prediction , acc