import re
import matplotlib
import sklearn.tree
from sklearn.datasets import load_iris
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier, plot_tree ,DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
# 导入分割数据集的方法
from sklearn.model_selection import train_test_split
# 导入科学计算包
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
import antropy as ant
from itertools import groupby

def normalize_array_to_range(arr, min_range=0, max_range=65535):  
    # 假设arr是一个包含(key, value)元组的列表  
    if not arr:  
        return []  
      
    # 找到value的最大值和最小值  
    min_value = min(value for _, value in arr)  
    max_value = max(value for _, value in arr)  
      
    # 如果最大值等于最小值，直接返回原数组（或全为min_range的数组）  
    if max_value == min_value:  
        return [(key, min_range) for key, value in arr]  
      
    # 归一化数组到(0, 65535)范围  
    normalized_arr = [(key, int((value - min_value) / (max_value - min_value) * (max_range - min_range) + min_range))  
                      for key, value in arr]  
      
    return normalized_arr  

def cal_runs(sarray):
    smedian=np.median(sarray)
    runtest=[]
    for i in sarray:
        if i<=smedian:
            runtest.append(0)
        else:
            runtest.append(1)
    return sum(1 for _ in groupby(runtest))

def duldiff(array):
    l=[]
    for i in range(len(array)//2):
        l.append(abs(array[2*i+1]-array[2*i]))
    return np.mean(abs(np.diff(l)))

def load_data(file):
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
                if ipid[i] == '-1':
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

            srun=cal_runs(sarray)
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
            mindiff=np.min(abs(np.diff(sarray)))
            dul=duldiff(sarray)
        
            #feature=[entropys,varx,meanxp,vars,dul,mindiff]  99.054%
            feature=[entropys,varx,meanxp,vars,entropyxp,varsp,means,entropyx,meanx,varxp,entropysp,meansp]
            data.append(feature)
            label.append(linelabel)
        else:
            break
    f.close()

    return data,label

def load_work_data(file):
    f=open(file, 'r', encoding='utf-8')
    data=[]
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
                if ipid[i] == '-1':
                    pass
                else:
                    linedata.append((i,int(ipid[i])))

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

            srun=cal_runs(sarray)
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
            mindiff=np.min(abs(np.diff(sarray)))
            dul=duldiff(sarray)
        
            #feature=[entropys,varx,meanxp,vars,dul,mindiff]  #99.054%
            feature=[entropys,varx,meanxp,vars,entropyxp,varsp,means,entropyx,meanx,varxp,entropysp,meansp]
            data.append(feature)
        else:
            break
    f.close()

    return data

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        

X_train,y_train = load_data('train.txt')     # the lines in file are 'ipv6_address [ipid] lable'
X_test,y_test = load_data('validation.txt')

Pre= load_work_data('R_0_1.txt')  #no label

def Mytest_criterion(my_criterion):
    # 创建决策时分类器-
    tree_model=DecisionTreeClassifier(criterion=my_criterion,max_depth=None,random_state=0,splitter="best")

    # 喂入数据
    tree_model.fit(X_train,y_train)
    #print(tree_model.score(X_test,y_test))

    g=0
    l=0
    r=0
    o=0

    predict_results=tree_model.predict(X_train)
    for predict_result in predict_results:
        if predict_result == 0:
            g+=1
        if predict_result == 1:
            l+=1
        if predict_result == 2:
            r+=1
        if predict_result == 3:
            o+=1

    '''draw_confusion_matrix(label_true=y_test,			# y_gt=[0,5,1,6,3,...]
                      label_pred=predict_result,	    # y_pred=[0,5,1,6,3,...]
                      label_name=["Global", "Local", "Random", "Odd"],
                      title="Confusion Matrix of IPID Validation",
                      pdf_save_path="Confusion_Matrix_on_Fer2013.jpg",
                      dpi=300)'''
    print(g,l,r,o)


    

   

Mytest_criterion("entropy")#信息熵