import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from collections import Counter
from sklearn import train_test_split

path = "C:\\Users\\mdkha\\Desktop\\MY stuff\\python\\ML\\wdbc.data"
df = pd.read_csv(path,sep=",",header=None,na_values=[0])
df=df.dropna(axis = 0)
label = (df[[1]] == "M").astype(int)
df=df.drop([0,1],axis=1)
df.columns =[f"feature{i}" for i in range(1,df.shape[1]+1)]
label.columns = ["result"]

x_train,x_test,y_train,y_test = train_test_split(df,label,test_size = 0.2 , random_state = 42,stratify = label)

plt.scatter(x_train[y_train == 0,0])