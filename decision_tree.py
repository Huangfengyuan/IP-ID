import re
import matplotlib
import sklearn.tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree ,DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import antropy as ant
from itertools import groupby
import random
import time

def normalize_array_to_range(arr, min_range=0, max_range=65535):  
    """normalize list

    Arguments:
        arr {list} -- list to be normalized
        min_range {int} -- minimum
        max_range {int} -- maximum
    Returns:
         -- normalized list
    """
    if not arr:  
        return []  
       
    min_value = min(value for _, value in arr)  
    max_value = max(value for _, value in arr)  
       
    if max_value == min_value:  
        return [(key, min_range) for key, value in arr]  
       
    normalized_arr = [(key, int((value - min_value) / (max_value - min_value) * (max_range - min_range) + min_range))  
                      for key, value in arr]  
      
    return normalized_arr  



def load_data(file):
    """calculate the feature of ipid sequences, using for train and validation

    Arguments:
        file -- each line contains an address, the ipid sequence and label, for example 2001::1234 [1,2,3,4,5,6,7,8,9,10] global

    Returns:
         data -- list contains features
         label -- corresponding label
    """
    f=open(file, 'r', encoding='utf-8')
    data=[]
    label=[]
    while True:
        line=f.readline()
        if line:
            linedata=[]
            linelabel=-1
            feature=[]
            try:
                addr=re.search(r'(.*?) \[',line).group().split()[0]
                ipid=((re.search(r'\[(.*?)\]',line).group())[1:-1]).split()
                clas=line.split()[-1]
            except:
                print('error '+addr)

            for i in range(len(ipid)):
                if ipid[i] == '-1' or ipid[i] == '0':
                    pass
                else:
                    linedata.append((i,int(ipid[i])))
            if clas == 'global':
                linelabel=0
            elif clas == 'local':
                linelabel=1
            elif clas == 'random':
                linelabel=2
            elif clas == 'odd':
                linelabel=3
            else:
                print(clas)
                print("classerror")

            sarray = []
            xarray = []
            yarray = []
            nlinedata=normalize_array_to_range(linedata)
            for i in nlinedata:
                sarray.append(i[1])
                if i[0]%2 == 0:
                    xarray.append(i[1])
                else:
                    yarray.append(i[1])
            sarray=np.array(sarray)
            xarray=np.array(xarray)
            yarray=np.array(yarray)

            entropys=ant.perm_entropy(sarray, normalize=False)
            varx=np.var(xarray)
            meanxp=np.mean(abs(np.diff(xarray)))
            means=np.mean(sarray)
            meanx=np.mean(xarray)
            meansp=np.mean(abs(np.diff(sarray)))
            vars=np.var(sarray)
            entropyxp=ant.perm_entropy(abs(np.diff(xarray)), normalize=False)
            entropysp=ant.perm_entropy(abs(np.diff(sarray)), normalize=False)
            entropyx=ant.perm_entropy(xarray, normalize=False)
            varsp=np.var(abs(np.diff(sarray)))
            varxp=np.var(abs(np.diff(xarray)))

            #feature=[entropys,varx,meanxp,vars,dul,mindiff]  99.054%
            feature=[entropys,varx,meanxp,vars,entropyxp,varsp,means,entropyx,meanx,varxp,entropysp,meansp]
            data.append(feature)
            label.append(linelabel)
        else:
            break
    f.close()

    return data,label

def load_work_data(file):
    """calculate the feature of ipid sequences, using for predicting

    Arguments:
        file -- each line contains an address, the ipid sequence , for example 2001::1234 [1,2,3,4,5,6,7,8,9,10]

    Returns:
         addresses -- ipv6 address
         data -- list contains features
    """
    f=open(file, 'r', encoding='utf-8')
    data=[]
    addresses=[]
    while True:
        line=f.readline()
        if line:
            linedata=[]
            feature=[]
            
            try:
                addr=re.search(r'(.*?) \[',line).group().split()[0]
                ipid=((re.search(r'\[(.*?)\]',line).group())[1:-1]).split()
                clas=line.split()[-1]
            except:
                print('error '+addr)

            for i in range(len(ipid)):
                if ipid[i] == '-1' or ipid[i] == '0':
                    pass
                else:
                    linedata.append((i,int(ipid[i])))
            if len(linedata)<20:
                continue
            sarray = []
            xarray = []
            yarray = []
            nlinedata=normalize_array_to_range(linedata)
            for i in nlinedata:
                sarray.append(i[1])
                if i[0]%2 == 0:
                    xarray.append(i[1])
                else:
                    yarray.append(i[1])
            sarray=np.array(sarray)
            xarray=np.array(xarray)
            yarray=np.array(yarray)

            entropys=ant.perm_entropy(sarray, normalize=False)
            varx=np.var(xarray)
            meanxp=np.mean(abs(np.diff(xarray)))
            means=np.mean(sarray)
            meanx=np.mean(xarray)
            meansp=np.mean(abs(np.diff(sarray)))
            vars=np.var(sarray)
            entropyxp=ant.perm_entropy(abs(np.diff(xarray)), normalize=False)
            entropysp=ant.perm_entropy(abs(np.diff(sarray)), normalize=False)
            entropyx=ant.perm_entropy(xarray, normalize=False)
            varsp=np.var(abs(np.diff(sarray)))
            varxp=np.var(abs(np.diff(xarray)))
        
            #feature=[entropys,varx,meanxp,vars,dul,mindiff]  #99.054%
            feature=[entropys,varx,meanxp,vars,entropyxp,varsp,means,entropyx,meanx,varxp,entropysp,meansp]
            data.append(feature)
            addresses.append(addr)
        else:
            break
    f.close()

    return addresses,data

def draw_confusion_matrix(label_true, label_pred, label_name):
    """
    drawing the confusion matrix of validation result

    Arguments:
        label_true: true label
        label_pred: predict label
        label_name: label name

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.xlabel("Predict label",fontsize=15)
    plt.ylabel("Truth label",fontsize=15)
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0) 
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    plt.show()
 

def Mytest_criterion(train_file,validation_file,prediction_file,my_criterion):
    """
    training, validating and predicting

    """
    X_train,y_train = load_data(train_file) 
    X_test,y_test = load_data(validation_file)

    tree_model=DecisionTreeClassifier(criterion=my_criterion,max_depth=None,random_state=0,splitter="best")
    tree_model.fit(X_train,y_train)
    
    score=tree_model.score(X_test,y_test)             #validation result

    addresses,Pre= load_work_data(prediction_file)
    predict_results=tree_model.predict(Pre)           #prediction result

train_file=''                      #data using for training, each line contains an address, the ipid sequence and label, for example 2001::1234 [1,2,3,4,5,6,7,8,9,10] global
validation_file=''                 #data using for validating, each line contains an address, the ipid sequence and label, for example 2001::1234 [1,2,3,4,5,6,7,8,9,10] global
prediction_file=''                 #data using for predicting, each line contains an address, the ipid sequence, for example 2001::1234 [1,2,3,4,5,6,7,8,9,10]

Mytest_criterion(train_file,validation_file,prediction_file,"entropy")
