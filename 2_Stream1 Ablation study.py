from textwrap import fill
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

plt.style.use('fivethirtyeight')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.ensemble import RandomForestClassifier

import numpy as np
import timeit

import torch

torch.manual_seed(42)

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#%%Load the heterogeneous graph data
cur_dir = os.getcwd()

data = torch.load(cur_dir+'/data/Vulnerability_hetero_graph_data_destination_balanced.pt')
print(data)
print(data['Vulnerability'].train_mask.sum())
print(data['Vulnerability'].test_mask.sum())

x_embedding = data['Vulnerability'].x.numpy()


#%% load stream 1 features
with open(cur_dir+'/data/SubGraph_balanced_metadata.pickle', 'rb') as f:
    SubGraph_balanced_metadata = pickle.load(f)
Vulnerability_features=SubGraph_balanced_metadata["Vulnerability_features"]

Features = ["Stream1: R1","Stream1: R2","Stream1: R1 + R2"]

for Feature in Features:
    if Feature=="Stream1: R1":
        x_AFFECTS=Vulnerability_features.iloc[:,5:13].to_numpy() #
        # Putting feature variable to X
    elif Feature == "Stream1: R2":
        x_AFFECTS = Vulnerability_features.iloc[:, 13:].to_numpy()  #
    elif Feature == "Stream1: R1 + R2":
        x_AFFECTS = Vulnerability_features.iloc[:, 5:].to_numpy()

    x=np.concatenate((x_embedding, x_AFFECTS), axis=1)
    # split the dataset
    train_mask=data['Vulnerability'].train_mask.numpy()
    test_mask=data['Vulnerability'].test_mask.numpy()

    X_train = x[train_mask]
    X_test = x[test_mask]

    # Putting response variable to y
    y_train = data['Vulnerability'].y[data['Vulnerability'].train_mask].numpy()
    y_test = data['Vulnerability'].y[data['Vulnerability'].test_mask].numpy()

    # Feature Scaling
    sc = StandardScaler()
    sc.fit_transform(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)


    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)

    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)



    #%% set parameters for results saving
    result_dir=cur_dir+'/data/ML_classification_performance.csv'
    plot_dir=result_dir.replace('.csv','_'+Feature+'ROC.npy') # prepare data for ROC curve

    with open(result_dir, 'a', newline='') as f:
        writer = csv.writer(f)
        my_list = ['Classifier',"Feature","trainTime(s)",'test_cm','test_acc','test_pre','test_rec','test_f1',
                   'test_class1_pre','test_class1_rec','test_class1_f1',
                   'test_class0_pre', 'test_class0_rec', 'test_class0_f1']
        writer.writerow(my_list)



    #%% build classifier and evaluate performance
    c_DT = DecisionTreeClassifier(max_depth=50)
    c_ABC = AdaBoostClassifier(base_estimator=c_DT, n_estimators=9)
    c_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
    c_MLP = MLPClassifier(random_state=1, max_iter=300)
    c_RF = RandomForestClassifier(n_jobs=-1)
    c_XGB= XGBClassifier()

    classifier_list = [c_ABC,c_DT,c_KNN,c_MLP,c_RF,c_XGB]
    classifier_name_list = ["c_ABC",'c_DT',"c_KNN",'c_MLP',"c_RF","c_XGB"]

    dic_draw= {} # prepare data for ROC curve
    for c in range(len(classifier_list)):
        classifier = classifier_list[c]
        classifier_name = classifier_name_list[c]

        # classifier train
        start = timeit.default_timer()
        classifier.fit(X_train, y_train)
        stop = timeit.default_timer()
        trainTime=stop-start
        # evaluate classifier on the test set
        predictions = classifier.predict(X_test)
        test_confusion_matrix = confusion_matrix(y_test, predictions)

        report = classification_report(y_test, predictions, labels=[0, 1], target_names=['class 0', 'class 1'],
                                       output_dict=True, zero_division=0)  # output_dict=True
        test_acc = report['accuracy']
        test_pre = report['macro avg']['precision']
        test_rec = report['macro avg']['recall']
        test_f1 = report['macro avg']['f1-score']

        test_class1_pre = report['class 1']['precision']
        test_class1_rec = report['class 1']['recall']
        test_class1_f1 = report['class 1']['f1-score']

        test_class0_pre = report['class 0']['precision']
        test_class0_rec = report['class 0']['recall']
        test_class0_f1 = report['class 0']['f1-score']

        # save the results
        with open(result_dir, 'a', newline='') as f:
            writer = csv.writer(f)
            my_list = [classifier_name,Feature, trainTime, test_confusion_matrix, test_acc, test_pre, test_rec, test_f1,
                       test_class1_pre, test_class1_rec, test_class1_f1,
                       test_class0_pre, test_class0_rec, test_class0_f1]
            writer.writerow(my_list)

        print([classifier_name,trainTime])
        print([test_acc, test_pre, test_rec, test_f1,
               test_class1_pre, test_class1_rec, test_class1_f1,
               test_class0_pre, test_class0_rec, test_class0_f1])

        # prepare data for ROC curve
        pos_score = classifier.predict_proba(X_test)[:, 1]
        fpr, tpr, threshold = roc_curve(y_test, pos_score)
        dic_draw.update({classifier_name: [fpr, tpr]})
    np.save(plot_dir,dic_draw) # prepare data for ROC curve



    #%% roc curve
    classifier_name_list = ["c_ABC",'c_DT',"c_KNN",'c_MLP',"c_RF","c_XGB"]
    color_list = ['red', 'black', 'blue', 'orange', 'brown', 'pink']
    leg_length = 15
    fsize = 22
    lw = 1.5
    # plt.figure(figsize=(16, 8))
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    fdict = {'weight': 2, 'fontsize': fsize}

    # plot all 8 figures:
    result_dir=cur_dir+'/data/ML_classification_performance.csv'
    data_plot = np.load(plot_dir, allow_pickle=True).item()

    for k in range(len(classifier_name_list)):
        print(classifier_name_list[k])

        [fpr, tpr] = data_plot[classifier_name_list[k]]
        roc_auc = auc(fpr, tpr)
        lab = classifier_name_list[k].replace("c_","")

        plt.plot(fpr, tpr, color=color_list[k], lw=lw, label=fill(lab + '(AUC=%0.3f)' % roc_auc, leg_length))

    plt.tick_params(labelsize=fsize - 2)
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=fsize)
    plt.ylabel('True Positive Rate', fontsize=fsize)
    plt.title('Random', x=0.8, y=0.65, fontdict=fdict)
    plt.legend(loc="lower right", fontsize=fsize)
    plt.tight_layout()
    plt.savefig(result_dir.replace('.csv','_'+Feature+'ROC.pdf'),dpi=600)
    plt.show()

