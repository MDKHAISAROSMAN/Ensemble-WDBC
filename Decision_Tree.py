import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def import_data(path):
    df = pd.read_csv(path,sep = ",",header = None)
    label = (df[1] == "M").astype(int)
    df = df.drop([0,1],axis = 1)
    df.columns = [f"feature{i}" for i in range (1,df.shape[1]+1)]
    label.columns = ["Result"]
    label = label.values
    return df,label

def split(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
    return x_train,x_test,y_train,y_test

class Node:
    def __init__(self,feature_idx=None,threshold=None,info_gain=None,left=None,right=None,value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.info_gain = info_gain
        self.left = left
        self.right = right
        self.value = value

class decision_tree:
    def __init__(self,min_samples_split = 2,max_depth = 2):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth


    def build_tree(self,dataset,curr_depth=0):
        x,y = dataset[:,:-1], dataset[:,-1]
        n_samples,n_feature = x.shape

        if n_samples>= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.best_split(dataset,n_feature)

            if best_split["info_gain"]>0:
                left_node = self.build_tree(best_split["left_dataset"],curr_depth+1)
                right_node = self.build_tree(best_split["right_dataset"],curr_depth+1)

                return Node(best_split["feature_idx"],best_split["threshold"],best_split["info_gain"],left_node,right_node)
        
        leaf_value = Counter(y).most_common(1)[0][0]
        return Node(value = leaf_value)


    def best_split(self,dataset,n_feature):
        best_split = {"feature_idx":None,"threshold":None,"info_gain":-1,"left_dataset":None, "right_dataset":None}
        
        for feature_idx in range (n_feature):
            feature_values = dataset[:,feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset , right_dataset = self.split(dataset,feature_idx,threshold )

                if len(left_dataset) and len(right_dataset):
                    parent_y,left_y,right_y = dataset[:,-1],left_dataset[:,-1],right_dataset[:,-1]

                    info_gain = self.information_gain(parent_y,left_y,right_y)

                    if info_gain > best_split["info_gain"]:
                        best_split["feature_idx"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["info_gain"] = info_gain
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
        
        return best_split


    def split(self,dataset,feature_idx,threshold):
        left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])
        right_dataset = np.array([row for row in dataset if row[feature_idx] > threshold])
        return left_dataset,right_dataset


    def information_gain(self,parent_y,left_y,right_y):
        left_weight = len(left_y) / len(parent_y)
        right_weight = len(right_y) / len(parent_y)

        information_gain = self.entropy(parent_y) - (left_weight * self.entropy(left_y) + right_weight * self.entropy(right_y))
        return information_gain


    def entropy(self,y):
        entropy = 0 

        class_labels = np.unique(y)
        for class_label in class_labels:
            p = len(y[y == class_label]) / len(y)
            entropy += -p * np.log2(p)
        return entropy


    def fit(self,x,y):
        dataset = np.concatenate([x,y.reshape(-1,1)],axis = 1)
        self.root = self.build_tree(dataset)


    def predict(self,x):
        predictions = [self.predict_class(row,self.root) for row in x]
        return predictions


    def predict_class(self,row,node):
        if node.value != None:
            return node.value
        
        feature_val = row[node.feature_idx]
        if feature_val <= node.threshold:
            return self.predict_class(row,node.left)
        else:
            return self.predict_class(row,node.right)


def Tree(path,test):
    df, label = import_data(path)
    x_train, x_test, y_train, y_test = split(df, label)
    dt = decision_tree(min_samples_split=2, max_depth=4)
    dt.fit(x_train, y_train)
    predictions = dt.predict(x_test.values)
    accuracy = np.mean(predictions == y_test) * 100

    if test is None:
        return accuracy
    else:
        test_prediction = dt.predict(test.reshape(1, -1))
        return test_prediction[0].astype(int), accuracy