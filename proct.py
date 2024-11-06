import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler           
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, roc_curve     
from sklearn.metrics import accuracy_score                 
from sklearn.metrics import f1_score                      
from sklearn.metrics import matthews_corrcoef              
from sklearn.metrics import confusion_matrix               
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)


def evaluating_indicator(y_test, pre_label, proba):
    c_m = confusion_matrix(y_test, pre_label)
    TP=c_m[1,1]
    FN=c_m[1,0]
    FP=c_m[0,1]
    TN=c_m[0,0]
    
    TPR=TP/ (TP+ FN) 
    TNR= TN / (FP + TN)
    BER=1/2*((FP / (FP + TN) )+FN/(FN+TP))
    
    ACC = accuracy_score(y_test, pre_label)
    MCC = matthews_corrcoef(y_test, pre_label)
    F1score = f1_score(y_test, pre_label)
    AUC = roc_auc_score(y_test, proba[:,1])
    KAPPA=kappa(c_m)
    
    c={"ACC": ACC, "AUC": AUC, "TPR": TPR, "TNR": TNR, "BER": BER, "MCC": MCC, "F1_score": F1score, 'KAPPA': KAPPA}
    
    return c


def blo(pro_model_Pre,jj):     
    blo_Pre=np.zeros(len(pro_model_Pre)) 
    blo_Pre[(pro_model_Pre[:,1]>(jj*0.01))]=1
    return blo_Pre


def spec_for_ser(df,row_id): 
    str_df=str(df)
    for i in row_id:
        if i==row_id[0]: 
            input_mulit=(str_df+"["+str_df+"['hadm_id']=={}]".format(i))
        else:
            input_mulit=(input_mulit+","+str_df+"["+str_df+"['hadm_id']=={}]".format(i))
    return (pd.concat(eval(input_mulit),axis=0,ignore_index=True))

all_parameters = pd.read_csv('E:\Project\炎症因子\inflammatory_factor_end.csv')

min_parameters = all_parameters.loc[:, ['hadm_id', 'MCP-4', 'CXCL12', 'IL-1RA', 'MRP8/14', 'Cystatin C', 'IP-10', 'Eotaxin', 'MCP-1', 'MIP-3α', 'IL-18', 'MODS']]
hadm_id = list(set(min_parameters['hadm_id']))
train_id, test_id = train_test_split(hadm_id, test_size=0.2, random_state=42)

scaler = StandardScaler()
X = min_parameters.drop(['hadm_id', 'MODS'], axis = 1, inplace = False)
X.iloc[:, :] = scaler.fit_transform(X)
Y = min_parameters.iloc[:, -1]

x_train = spec_for_ser('min_parameters', train_id)
y_train = x_train.iloc[:, -1]
x_train.drop(['hadm_id', 'MODS'], axis = 1, inplace = True)
x_train.iloc[:, :] = scaler.fit_transform(x_train)

x_test = spec_for_ser('min_parameters', test_id)
y_test = x_test.iloc[:, -1]
x_test.drop(['hadm_id', 'MODS'], axis = 1, inplace = True)
x_test.iloc[:, :] = scaler.fit_transform(x_test)

params = {                      
    'boosting_type': 'gbdt',   
    'objective': 'binary',      
    'metric': 'binary_error',   
    'learning_rate': 0.05,      
    'n_estimators': 200,         
    'max_depth': 100,            
    'num_leaves': 16,         
    'subsample': 0.7,           
    'reg_alpha': 0.01,       
    'reg_lambda': 1.0,        
    'is_unbalance': True
}
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

model = DecisionTreeClassifier(max_depth=params['max_depth'], class_weight='balanced')
param = {'C': 10, 'gamma': 0.003, 'kernel': 'sigmoid'}
model.fit(x_train, y_train)
pre_proba = model.predict_proba(x_test)
RightIndex=[]
for jj in range(100):                   
    blo_pre = blo(pre_proba,jj)
    eva_index = evaluating_indicator(y_test, blo_pre, pre_proba)
    RightIndex.append(abs(eva_index['TPR'] - eva_index['TNR']))
RightIndex=np.array(RightIndex,dtype=np.float16)
position=np.argmin(RightIndex)       
position=position.mean()

final_blo_pre = blo(pre_proba, position) 

evaluation_index = evaluating_indicator(y_test, final_blo_pre, pre_proba)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
ACC = []
AUC = []
TPR = []
TNR = []
BER = []
MCC = []
F1_score = []
KAPPA = []


for train_index, test_index in kf.split(min_parameters):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = Y[train_index], Y[test_index]
    model = DecisionTreeClassifier(max_depth=params['max_depth'], class_weight='balanced')
    model.fit(X_train, y_train)
    pre_proba = model.predict_proba(X_test)
    RightIndex=[]
    for jj in range(100):                   
        blo_pre = blo(pre_proba,jj)
        eva_index = evaluating_indicator(y_test, blo_pre, pre_proba)
        RightIndex.append(abs(eva_index['TPR'] - eva_index['TNR']))
    RightIndex=np.array(RightIndex,dtype=np.float16)
    position=np.argmin(RightIndex)         
    position=position.mean()
    probs = pre_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(roc_auc)
    
    final_blo_pre = blo(pre_proba, position) 

    evaluation_index = evaluating_indicator(y_test, final_blo_pre, pre_proba)
    
    ACC.append(evaluation_index['ACC'])
    AUC.append(evaluation_index['AUC'])
    TPR.append(evaluation_index['TPR'])
    TNR.append(evaluation_index['TNR'])
    BER.append(evaluation_index['BER'])
    MCC.append(evaluation_index['MCC'])
    F1_score.append(evaluation_index['F1_score'])
    KAPPA.append(evaluation_index['KAPPA'])


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.figure()
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

tprs_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
tprs_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
