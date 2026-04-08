from liner_regression import regression
from Decision_Tree import Tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from knn import K_NN

def main():
    path = "C:\\Users\\mdkha\\Desktop\\MY stuff\\python\\ML\\wdbc.data"
    #path = input("Enter path: ")
    test_vector = np.array([9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773])
    
    knn_prediction , knn_accuracy = K_NN("test",path,test_vector)
    print(knn_prediction,knn_accuracy,sep="---Knn---")
    
    LR_prediction , LR_accuracy = regression("test",path,test_vector)
    print(LR_prediction,LR_accuracy,sep = "---LR---")
    
    DT_prediction,DT_accuracy = Tree(path,test_vector)
    print (DT_prediction,DT_accuracy,sep = "---DT---")
main()