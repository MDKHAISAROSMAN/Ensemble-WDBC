import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(file_path):
    df = pd.read_csv(file_path,sep=",",header=None,na_values=[0])
    df=df.dropna(axis = 0)
    label = (df[[1]] == "M").astype(int)
    df=df.drop([0,1],axis=1)
    df.columns =[f"feature{i}" for i in range(1,df.shape[1]+1)]
    label.columns = ["result"]
    return df,label


def split_data(df,label):
    x_train,x_test,y_train,y_test = train_test_split(df,label,random_state=42,stratify=label,test_size=0.25)
    return x_train,x_test,y_train,y_test


def normalize(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    mean = mean.values.reshape(1,-1)
    std = np.std(x_train, axis=0)
    std = std.values.reshape(1,-1)
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std
    return x_train, x_test,mean,std


def train(x_train,y_train,epoch,alpha):
    m = y_train.shape[0]
    b = np.random.uniform(0,1,(1,1))
    w = np.random.uniform(0,1,(30,1))
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


def train_accuracy(x,y,w,b):
    z= (x @ w) + b
    A = 1/(1+np.exp(-z))
    y_pred = (A >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    print("Training Accuracy:", accuracy*100)


def test_accuracy(x,y,w,b):
    z = (x @ w)+b
    K = 1/(1+np.exp(-z))
    y_pred = (K>=0.5).astype(int)
    accuracy = np.mean(y_pred == y) * 100
    print(f"Accuracy of linear regression is: {accuracy}")


def predict(test_input,w,b):
    z = (test_input @ w)+b
    K = 1/(1+np.exp(-z))
    y_pred = (K>=0.5).astype(int)
    print(y_pred)


def save(w,b,mean,std):
    np.savez("model.npz",w=w,b=b,mean=mean,std=std)

def linear():
    path = "C:\\Users\\mdkha\\Desktop\\MY stuff\\python\\ML\\wdbc.data"
    alpha = 0.5
    epoch = 200
    a = np.array([11.64,18.33,75.17,412.5,0.1142,0.1017,0.0707,0.03485,0.1801,0.0652,0.306,1.657,2.155,20.62,0.00854,0.0231,0.02945,0.01398,0.01565,0.00384,13.14,29.26,85.51,521.7,0.1688,0.266,0.2873,0.1218,0.2806,0.09097])
    a = a.reshape(1,-1)
    df , label =load_data(path)
    x_train,x_test,y_train,y_test = split_data(df,label)
    x_train,x_test,mean,std = normalize(x_train,x_test)
    w,b = train(x_train,y_train,epoch,alpha)
    save(w,b,mean,std)
    train_accuracy(x_train,y_train,w,b)
    test_accuracy(x_test,y_test,w,b)

linear()