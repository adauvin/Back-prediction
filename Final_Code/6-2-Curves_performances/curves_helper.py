
# coding: utf-8

# # 1 - Description

# The purpose of this file is to plot all the curves created for the predicted probabilities of the model on the test set (ROC, PR, Calibration, Confusion matrix).
# 
# 
# ### INPUT
# - **probabilities_with_'+str(number_of_patients) +'_patients_test_size_'+str(test_size)**/ **Probabibilities** - A pickle file being a dictionary containing all the predicted probabilities on the test set for every model, target and seed.
# 
# 
# ### OUTPUT
# 
# 
# - All the curves mentionned as figures in the Results/figures/Curves folder
# 
# 
# ----------------------------------------------------------
# 

# # 1- Packages Import

# In[11]:

# Basic packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns



import pickle

# Handling datetime types
from datetime import datetime

# Imputation

import sklearn.preprocessing.imputation
from sklearn.preprocessing import Imputer
# from sklearn.impute import SimpleImputer

# Split
from sklearn.model_selection import train_test_split

#Models
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


from sklearn.metrics import classification_report,confusion_matrix
import itertools

import matplotlib.pyplot as plt



path_to_load='../../Data/Generated_csv/'
imputation_method='fillna'

# ## 2-2 Load the results initialize path


path='../../Data/Results/'

creatinine=False
path_to_add='Hemoglobin/'
if creatinine:
    path_to_add='Creatinine/'
path='../../Data/Results/'+path_to_add

# # 2- Functions

# ## 2-1 Plots all curves

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator.
# This ensures that the results below are reproducible.
dict_models={'RFC':'Random Forest','XGB': 'Gradient Boosted Trees', 'Log': 'Logistic Regression','Lin': 'Linear Regression','CART':'Decision Tree','OCT': 'Optimal Classification Trees','ORT': 'Optimal Regressor Trees','Baseline':'Baseline' }



def plot_bland(true_y,
               preds,
               target,
               model_name,
               seed,
               number_of_patients,
               is_creatinine=False,
               is_filter=False,
               save_img=False
               ):
    f, ax = plt.subplots(1, figsize = (8,5))
    
    sm.graphics.mean_diff_plot(preds,true_y,  ax = ax)
    title_to_add= "Hemoglobin"
    if is_creatinine:
        title_to_add= "Creatinine"
    
    if is_filter:
        title_to_add+=", patients with abnormal admissions results only"

    target_dict={"target_baseline":"Continuous prediction"}
    plt.title("Bland-Altman Plot': "+dict_models[model_name]+' ' +target_dict[target]+' for '+title_to_add)
    #plt.show()

    
def plot_residuals(true_y,
               preds,
               target,
               model_name,
               seed,
               number_of_patients,
               is_creatinine=False,
               is_filter=False,
               save_img=False
               ):
    
    plt.figure()
    
    
    BLcolor = 'maroon'
    
    if is_creatinine:
        BLcolor = 'orange'
    Admcolor = 'gray'
    Opacity = 0.8

    diff = true_y-preds
    diff =pd.DataFrame(diff)
    plt.hist(diff[0], stacked=True, color = BLcolor, alpha=Opacity, bins = 60, range=(-5,6))
    
    if is_creatinine:
        plt.xlabel('Error in creatinine prediction (mg/dL)')
    if not is_creatinine:
        plt.xlabel('Error in hemoglobin prediction (g/dL)')

    plt.ylabel('Frequency (number of patients)')
    #plt.title('Hemoglobin: Histogram of the residuals')
    plt.xticks(np.arange(-5, 6, 1.0))
    

    
    title_to_add= "Hemoglobin"
    if is_creatinine:
        title_to_add= "Creatinine"
    
    if is_filter:
        title_to_add+=", patients with abnormal admissions results only"
    
    target_dict={"target_baseline":"Continuous prediction"}
    plt.title("Histogram of the residuals: "+dict_models[model_name]+' ' +target_dict[target]+' for '+title_to_add)
#plt.show()








# ### 2-1-1 Plot ROC


from sklearn.metrics import roc_curve

def plot_ROC(true_y,
             preds,
             target,
             model_name,
             seed,
             number_of_patients,
             colors={'RFC':'m','XGB':'r','Log':'y','CART':'g','OCT':'b','Baseline':'black'},
             on_same_graph=False,
             is_creatinine=False,
             is_filter=False,
             save_img=False
             ):
    
    
    BLcolor = colors[model_name]
    Admcolor = 'gray'

    Opacity = 0.8
    
    title_to_add= "Hemoglobin"
    if is_creatinine:
        title_to_add= "Creatinine"
    
    if is_filter:
        title_to_add+=", patients with abnormal admissions results only"

    AUC_test = roc_auc_score(true_y,preds)
    print("AUC ROC on test set "+model_name+": ",AUC_test)
    
    fpr,tpr, thresholds = roc_curve(true_y,preds)
    
    if not on_same_graph:
        plt.figure(figsize=(5, 5))
    plt.plot(fpr,tpr, color = BLcolor, alpha=Opacity)

    plt.xlim([0,1])
    plt.ylim([0,1])
    if not on_same_graph:
        plt.title(model_name+' '+target+' prediction for '+str(number_of_patients)+' patients , AUC ROC curve')

    target_dict={"target_bin":"Binary prediction","target_bin2":"Binary prediction for less than 8","target_AKI": "AKI prediction"}
    if on_same_graph:
        plt.title("ROC curve: " +target_dict[target]+' for '+title_to_add)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(False)
    
    if save_img:
        if on_same_graph:
            plt.savefig(path+'/figures/Curves/ROC_'+target+'_with'+str(number_of_patients)+'_patients_seed'+str(seed),dpi=400,bbox_inches="tight")
        if not on_same_graph:
            plt.savefig(path+'/figures/Curves/ROC_'+target+'for_'+model_name+'_with'+str(number_of_patients)+'_patients_seed'+str(seed),dpi=400,bbox_inches="tight")




#### 2-1-1 Plot PR

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature

from sklearn.metrics import average_precision_score

def plot_PR(true_y,
            preds,
            target,
            model_name,
            seed,
            number_of_patients,
            colors={'RFC':'m','XGB':'r','Log':'y','CART':'g','OCT':'b','Baseline':'black'},
            on_same_graph=False,
            is_creatinine=False,
            is_filter=False,
            save_img=False
            ):
    
    

    title_to_add= "Hemoglobin"
    if is_creatinine:
        if model_name=='Baseline':
            for i in range(len(preds)):
                preds[i]=0
        
        title_to_add= "Creatinine"
    
    if is_filter:
        title_to_add+=", patients with abnormal admissions results only"

    average_precision = average_precision_score(true_y,preds)

    print(model_name+' Average precision-recall score: {0:0.2f}'.format(average_precision))
    
    precision, recall, thresholds2 = precision_recall_curve(true_y,preds)
    if not on_same_graph:
        plt.figure(figsize=(5, 5))
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
    if 'step' in signature(plt.fill_between).parameters
    else {})
    plt.step(recall, precision, color=colors[model_name], alpha=0.8,where='post')
    #plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    if not on_same_graph:
        plt.title(model_name+' '+target+' prediction for '+str(number_of_patients)+' patients ,Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    target_dict={"target_bin":"Binary prediction","target_bin2":"Binary prediction baseline <8","target_AKI": "AKI prediction"}
    if on_same_graph:
        plt.title("PR curve: " +target_dict[target]+' for '+title_to_add)

    if save_img:
        if on_same_graph:
            plt.savefig(path+'/figures/Curves/PR_for_'+target+'_with'+str(number_of_patients)+'_patients_seed'+str(seed),dpi=400,bbox_inches="tight")
        if not on_same_graph:
            plt.savefig(path+'/figures/Curves/PR_for_'+target+'for_'+model_name+'_with'+str(number_of_patients)+'_patients_seed'+str(seed),dpi=400,bbox_inches="tight")




# ### 2-1-1 Plot Calibrations plot


from sklearn.calibration import calibration_curve
def plot_Calibration(true_y,
                     preds,
                     target,
                     model_name,
                     seed,
                     number_of_patients,
                     save_img=False
                     ):
    prob_true,prob_pred = calibration_curve(true_y,preds, normalize=False, n_bins=20)
    plt.figure(figsize=(5,5))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    name='RFC'
    ax1.plot(prob_pred,prob_true, "s-",label="%s" % (name, ))
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    if save_img:
        plt.savefig(path+'/figures/Curves/Calibration_for_'+target+'for_'+model_name+'_with'+str(number_of_patients)+'_patients_seed'+str(seed),dpi=400,bbox_inches="tight")


# ### 2-1-4 Plot Confusion Matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 3
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




#change that after plane
# actually fine for now

def get_recall_index(recall,min_precision_or_recall):
    index=0
    for t in recall:
        if t<min_precision_or_recall:
            return index-1
        index+=1
    return index

def get_precision_index(recall,min_precision_or_recall):
    index=0
    for t in recall:
        if t>min_precision_or_recall:
            return index-1
        index+=1
    return index

###

def plot_Confusion_matrix(true_y,
                          preds,
                          target,
                          model_name,
                          seed,
                          number_of_patients,
                          min_precision_or_recall,
                          focus_on_recall=True,
                          save_img=False):

    precision, recall, thresholds = precision_recall_curve(true_y,preds)
    if focus_on_recall:
        index = get_recall_index(recall,min_precision_or_recall)
    if not focus_on_recall:
        index = get_precision_index(precision,min_precision_or_recall)
    threshold=thresholds[index]
    preds_bin = preds>threshold

    print("AUC : ",round(roc_auc_score(true_y,preds),3))
    print(classification_report(true_y,preds_bin))

    plot_confusion_matrix(cm           = np.array(confusion_matrix(true_y,preds_bin)), 
                          normalize    = False,
                          target_names = ['Is a 0', 'Is a 1'],
                          title        = 'Confusion Matrix for '+target+' for '+model_name+' with recall '+str(round(recall[index],2))+' and precision '+str(round(precision[index],2))+' with '+str(number_of_patients)+' patients')
    
    if save_img:
        plt.savefig(path+'/figures/Curves/Confusion_Matrix_for_'+target+'for_'+model_name+'_with'+str(number_of_patients)+'_patients_seed'+str(seed),dpi=400,bbox_inches="tight")




# ## 2-2 Main functions

def get_curves_for_target(
    target,
    model_name,
    seed,
    number_of_patients,
    test_size=0.25,
    imputation_method='knn',
    curves=['AUC','PR','Calibration'],
    is_creatinine=False,
    save_img=False):


    path_to_add='Hemoglobin/'
    if is_creatinine:
        path_to_add='Creatinine/'
    path='../../Data/Results/'+path_to_add


    with open(path+'Probabilities/probabilities_with_'+str(number_of_patients) +'_patients_test_size_'+str(test_size)+imputation_method,'rb') as f:
        Probabilities=pickle.load(f)
    
    true_y = Probabilities[target][model_name][seed][target].values
    preds = Probabilities[target][model_name][seed]['proba_'+target].values
    
    if 'ROC' in curves:
        plot_ROC(true_y,preds,target,model_name,seed,number_of_patients,save_img=save_img)
    if 'PR' in curves:
        plot_PR(true_y,preds,target,model_name,seed,number_of_patients,save_img=save_img)
    if 'Calibration' in curves:
        plot_Calibration(true_y,preds,target,model_name,seed,number_of_patients,save_img=save_img)
        
    if 'Confusion Matrix' in curves:
        plot_Confusion_matrix(true_y,preds,target,model_name,seed,number_of_patients,min_precision_or_recall=0.8,focus_on_recall=True,save_img=False)




def save_all_curves(target_to_test,
                    model_name,
                    seed,
                    number_of_patients,
                    curves,
                    test_size=0.3,
                    save_img=True):
    
    for target in target_to_test:
        get_curves_for_target(
            target,
            model_name,
            seed,
            number_of_patients,
            test_size=0.3,
            curves=curves,
            save_img=save_img)


# All model one curve
from sklearn.metrics import roc_curve

def plot_all_models_one_graph(
                              target,
                              models_to_test,
                              seed,
                              number_of_patients,
                              imputation_method='mean',
                              test_size=0.25,
                              curve='ROC',
                              is_creatinine=False,
                              is_filter=False,
                              save_img=False
                              ):
    colors={'RFC':'m','XGB':'r','OCT':'deepskyblue','Log':'y','CART':'g','Baseline':'black'}

    path='../../Data/Results/'


    
    path_to_add='Hemoglobin/'
    if is_creatinine:
        path_to_add='Creatinine/'
    path='../../Data/Results/'+path_to_add

    filter=''
    if is_filter:
        filter='filtered'
    
    with open(path+'Probabilities/probabilities_with_'+str(number_of_patients) +'_patients_test_size_'+str(test_size)+imputation_method,'rb') as f:
        Probabilities=pickle.load(f)
    
    #print(Probabilities)
    
    plt.figure(figsize=(10, 10))
    for model_name in models_to_test:
        true_y = Probabilities[target][model_name][seed][target].values
        preds = Probabilities[target][model_name][seed]['proba_'+target].values
        if curve=='ROC':
            plot_ROC(true_y,preds,target,model_name,seed,number_of_patients,colors,on_same_graph=True,is_creatinine=is_creatinine, is_filter=is_filter,save_img=False)
        if curve=='PR':
            plot_PR(true_y,preds,target,model_name,seed,number_of_patients,colors,on_same_graph=True,is_creatinine=is_creatinine, is_filter=is_filter,save_img=False)
    plt.legend(models_to_test)
    if save_img:
        if curve=='ROC':
            plt.savefig(path+'figures/Curves/ROC_for_'+target+'_with'+str(number_of_patients)+'_patients_seed'+str(seed)+filter+'.png',dpi=400,bbox_inches="tight")
        elif curve=='PR':
            plt.savefig(path+'figures/Curves/PR_for_'+target+'_with'+str(number_of_patients)+'_patients_seed'+str(seed)+filter+'.png',dpi=400,bbox_inches="tight")

    plt.show()


def plot_continuous(target,
                  model_name,
                  seed,
                  number_of_patients,
                  imputation_method='knn',
                  test_size=0.25,
                  curve='Bland',
                  is_creatinine=False,
                  is_filter=False,
                  save_img=False
                  ):

    path='../../Data/Results/'
    
    
    
    path_to_add='Hemoglobin/'
    if is_creatinine:
        path_to_add='Creatinine/'
    path='../../Data/Results/'+path_to_add

    filter=''
    if is_filter:
        filter='filtered'

    with open(path+'Probabilities/probabilities_with_'+str(number_of_patients) +'_patients_test_size_'+str(test_size)+imputation_method+'_continuous','rb') as f:
        Probabilities=pickle.load(f)
    
    #print(Probabilities)
    
    plt.figure(figsize=(10, 10))
    true_y = Probabilities[target][model_name][seed][target].values
    preds = Probabilities[target][model_name][seed]['proba_'+target].values
    if curve=='Bland':
        plot_bland(true_y,preds,target,model_name,seed,number_of_patients,is_creatinine=is_creatinine, is_filter=is_filter)
    if curve=='Residual':
        plot_residuals(true_y,preds,target,model_name,seed,number_of_patients,is_creatinine=is_creatinine, is_filter=is_filter)
    if save_img:
        if curve=='Bland':
            plt.savefig(path+'figures/Curves/Bland_for_'+target+'_and_'+model_name+'_with'+str(number_of_patients)+'_patients_seed'+str(seed)+filter+'.png',dpi=400,bbox_inches="tight")
        elif curve=='Residual':
            plt.savefig(path+'figures/Curves/residuals_for_'+target+'_and_'+model_name+'_with'+str(number_of_patients)+'_patients_seed'+str(seed)+filter+'.png',dpi=400,bbox_inches="tight")
    plt.show()


