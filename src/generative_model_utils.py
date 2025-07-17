import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import random
import json
import torch 
from torch import nn
from joblib import Parallel, delayed
from datetime import datetime
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, precision_score,  balanced_accuracy_score, f1_score, roc_curve, roc_auc_score
from classifiers.mlp import TorchMLPClassifier
from lightgbm import LGBMClassifier

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
def create_folder(results_dire):    
    # create results dire
    if not os.path.exists(results_dire):
        os.makedirs(results_dire)
    
    else:
        warnings.warn('Folder already exists! Results will be overwritten!', category=Warning)
        
lr_model_args = {
            "random_state": SEED,
            "n_jobs": -1,
            "max_iter": 10000,
            "penalty": 'l2',
        }


svm_model_args = {
    "random_state": SEED,
    "max_iter": 10000,
}

rf_model_args = {
    "random_state": SEED,
    "n_jobs": -1,
    "n_estimators": 100,  
    "max_depth": None,  
}

mlp_model_args = {
    "random_state": SEED,
    "max_iter": 1000,
    "hidden_layer_sizes": (100,), 
    "activation": 'relu',  
    "solver": 'adam',
}



def Classifiers(X_train, y_train, X_test, y_test,  description , resulst_dire, name= None, save = True, detection = False, cm = False):
        # models = {
        # 'Logistic Regression': LogisticRegression(**lr_model_args),
        # #'SVM': SVC(**svm_model_args),
        # 'Random Forest': RandomForestClassifier(**rf_model_args),
        # 'MLP': MLPClassifier(**mlp_model_args)
        # }
        models = {
            'Logistic Regression': TorchMLPClassifier(hidden_dims=[], num_epochs=100, random_state=SEED, batch_size=256, verbose=True),
            'MLP': TorchMLPClassifier(hidden_dims=[100,], num_epochs=100, random_state=SEED, batch_size=256, verbose=True),
            'Random Forest': LGBMClassifier(boosting_type='rf', n_estimators=100, min_child_samples=2, colsample_bytree=0.01, random_state=SEED, verbose=-1)
        }
        

        labels = np.unique(y_test)
        results = {}
        
        for model_name, model in models.items():
            print(f'Training {model_name}')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            y_scores = model.predict_proba(X_test)[:, 1]
            labels_p = np.unique(y_pred)
            
            #print([x for x in labels_p if x not in labels])
            
            
            
            #description = f"{model_name}"
            df_result = show_single_class_evaluation(
            y_pred, y_test, y_scores, labels, name, resulst_dire, save=save, description=description, detection = detection 
            )
        
            results[model_name] = df_result

      
        return results        

def tissues_classification(data_real,tissue_label_real, data_gen ,tissue_label):
 
            
    
        cl5_r_vs_f = Classifiers(data_real, tissue_label_real , data_gen, tissue_label, None, None, save = False, detection = False, cm=True)
        print("tissues_evaluation")
        results = {}
        for model_name in cl5_r_vs_f.keys():
            results[model_name] = {}
 
            results[model_name]['balanced accuracy'] =  cl5_r_vs_f[model_name]['balanced_accuracy']
            results[model_name]['accuracy'] =  cl5_r_vs_f[model_name]['accuracy']
            results[model_name]['f1_weighted'] = cl5_r_vs_f[model_name]['f1_weighted']
            results[model_name]['f1'] = cl5_r_vs_f[model_name]['f1_macro']

        print(results)  
        return results 
    
    

def show_single_class_evaluation(y_pred: int, y_test: int, y_scores, labels, name=None, results_dire=None, verbose=False, save =True, description='', detection = False):
    
    dic_result = {} 
     
    if detection:
        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # print("Confusion Matrix:")
        # print(cm)
        # print(f"TP={tp}, FN={fn}, TN={tn}, FP={fp}")
        # print(f"Totali: Positivi={tp+fn}, Negativi={tn+fp}")

        # # Metriche base
        # print("\n--- Metriche base ---")
        # print("Balanced accuracy: ", round(balanced_accuracy_score(y_test, y_pred), 5))
        # print("Accuracy: ", round(accuracy_score(y_test, y_pred), 5))

        # # Macro e weighted (media delle classi)
        # print("\n--- Medie (macro e weighted) ---")
        # # print('Precision (macro): ', round(precision_score(y_test, y_pred, average="macro"), 5))
        # # print('Recall (macro): ', round(recall_score(y_test, y_pred, average="macro"), 5))
        # # print('F1 (macro): ', round(f1_score(y_test, y_pred, average="macro"), 5))
        # # print('F1 (weighted): ', round(f1_score(y_test, y_pred, average="weighted"), 5))
        # print('Precision (binary): ', round(precision_score(y_test, y_pred, average="binary"), 5))
        # print('Recall (binary): ', round(recall_score(y_test, y_pred, average="binary"), 5))
        # print('F1 (binary): ', round(f1_score(y_test, y_pred, average="binary"), 5))
        # ROC e AUC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = roc_auc_score(y_test, y_scores)
        #print("FPR:", np.round(fpr, 5))
        #print("TPR:", np.round(tpr, 5))
        # print("ROC AUC:", round(roc_auc, 5))


        # Metriche per classe
       # print("\n--- Per classe ---")
        # print("Precision: ", [round(i, 5) for i in precision_score(y_test, y_pred, average=None)])
        # print("Recall: ",  [round(i, 5) for i in recall_score(y_test, y_pred, average=None)])
        # print("F1 Score: ", [round(i, 5) for i in f1_score(y_test, y_pred, average=None)])


        dic_result['auc'] = [round(roc_auc,5)]
        
      
        dic_result['balanced_accuracy'] = [round(balanced_accuracy_score(y_test, y_pred), 5)]
        dic_result['accuracy'] = [round(accuracy_score(y_test, y_pred), 5)]
        dic_result['precision'] = [round(precision_score(y_test, y_pred, average="binary"), 5)]
        dic_result['recall'] = [round(recall_score(y_test, y_pred, average="binary"), 5)]
        dic_result['f1_macro'] = [round(f1_score(y_test, y_pred, average="binary"),5)]
        dic_result['f1_weighted'] = [round(f1_score(y_test, y_pred, average="binary"),5)]
        
        print('--------------------------------------------')
    
    else: 

    
        #roc_auc = roc_auc_score(y_test, y_scores)
        dic_result['balanced_accuracy'] = [round(balanced_accuracy_score(y_test, y_pred), 5)]
        dic_result['accuracy'] = [round(accuracy_score(y_test, y_pred), 5)]
        dic_result['precision'] = [round(precision_score(y_test, y_pred, average="macro"), 5)]
        dic_result['recall'] = [round(recall_score(y_test, y_pred, average="macro"), 5)]
        dic_result['f1_macro'] = [round(f1_score(y_test, y_pred, average="macro"),5)]
        dic_result['f1_weighted'] = [round(f1_score(y_test, y_pred, average="weighted"),5)]
        #dic_result['auc'] = [round(roc_auc,5)]
    


    
    for i in range(len(labels)):
        dic_result[str(labels[i])+'-precision'] =  round(precision_score(y_test, y_pred, average=None)[i], 5)
    for i in range(len(labels)):
        dic_result[str(labels[i])+'-recall'] =  round(recall_score(y_test, y_pred, average=None)[i], 5)
    for i in range(len(labels)):   
        dic_result[str(labels[i])+'-f1_score'] =  round( f1_score(y_test, y_pred, average=None)[i], 5)

    #df_result = pd.DataFrame.from_dict(dic_result)
    #df_result.to_csv(os.path.join(results_dire, name +'_output_detailed_scores.csv'), index=False)
    if save:
        #save_dictionary(dic_result, os.path.join(results_dire, name +'_output_detailed_scores',description))
        pass
    return dic_result

def save_dictionary(dictionary,filename,description):
    with open(filename + ".json", "w") as fp:
        fp.write("\n")
        fp.write(description)
        fp.write("\n")
        fp.write("\n")
        fp.write("\n")
        json.dump(dictionary, fp)
       
        print("Done writing dict into .json file")


def detection(data_real, data_gen, data_real_test, data_fake_test):
    #data_real_test, data_fake_test, cancer_label_test, tissue_label_test= self.generate_samples(real_test)
            
    detection_train_data = shuffle(np.vstack([data_real, data_gen]), random_state= SEED)
    detection_train_labels =shuffle(np.array([0] * len(data_real) + [1] * len(data_gen)), random_state=SEED)      
    detection_test_data = shuffle(np.vstack([data_real_test, data_fake_test]),random_state= SEED)
    detection_test_labels =shuffle(np.array([0] * len(data_real_test) + [1] * len(data_fake_test)), random_state=SEED)
    cl5_r_vs_f = Classifiers(detection_train_data, detection_train_labels , detection_test_data, detection_test_labels, None, None, save = False, detection = True)

    results = {}
    for model_name in cl5_r_vs_f.keys():
        results[model_name] = {}
        results[model_name]['accuracy'] =  cl5_r_vs_f[model_name]['accuracy']
        results[model_name]['f1'] = cl5_r_vs_f[model_name]['f1_macro']
        results[model_name]['auc'] = cl5_r_vs_f[model_name]['auc']
    print(results)  
    return results 