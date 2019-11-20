


# # 1- Packages Import

# In[1]:
from sklearn.metrics import average_precision_score
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


imputation_method='knn'


dict_models={'RFC':'Random Forest','XGB': 'Gradient Boosted Trees', 'Log': 'Logistic Regression','CART':'Decision Tree','OCT': 'Optimal Classification Trees' }

def get_target_AUC_boxplots(
                              target,
                              models_to_test,
                              seeds,
                              number_of_patients,
                              test_size=0.3,
                              save_img=False,
                              future=''):
    #initialize list to make data frame
    models_list = []
    seeds_list = []
    AUC_list = []
    
    constant_end_of_path=str(number_of_patients)+'_patients_test_size_'+str(test_size)+future
    
    #get AUC results from Bootstrapping
    with open(path+'AUC_results/auc_test_all_with_'+constant_end_of_path,'rb') as f:
        AUC_test=pickle.load(f)
    
    
    #fill the lists to create df
    for seed in seeds:
            for model_name in models_to_test:
                models_list.append(dict_models[model_name])
                seeds_list.append(seed)
                AUC_list.append(AUC_test[target][model_name][seed])
    df =pd.DataFrame([models_list, seeds_list, AUC_list]).T
    df.columns = ['models', 'seeds', 'AUC']

    df['AUC'] = df['AUC'].astype(float)


    plt.rcParams.update({'font.size': 12})
#plt.xlabel('xlabel', fontsize=10)
#plt.ylabel('ylabel', fontsize=16)
    f, ax2 = plt.subplots(1, 1, figsize=(14, 10))
    plt.margins(x=0.03)
    #two figures
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
#plt.title("Prediction Task "+ target)
    #sp = sns.stripplot(x='models', y="AUC", hue='models', data=df, ax=ax1)

    bp = sns.boxplot(x='models', y='AUC', hue='models',data=df, palette="Set2",ax=ax2)
    if save_img:
        plt.savefig(path+'/figures/AUC_box_plots/'+target+'box_plots_all_models'+constant_end_of_path+'.png',dpi=400,bbox_inches="tight",format='png')


# ## 2-2 Save AUC

# ### 2-2-3 Compute CI

# In[11]:

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return round(m-h,3), round(m+h,3)

def confidence_interval_width(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return 2*h




def get_or_save_PR_csv(
                        targets_to_test,
                        models_to_test,
                        seeds,
                        number_of_patients,
                        test_size=0.25,
                        save_AUC=False,
                        confidence=0.95,
                        is_Creatinine=False,
                        all_CI=False):
    
    
    

    path_to_add='Hemoglobin/'
    creatinine='_hemoglobin'
 
    if is_Creatinine:
        path_to_add='Creatinine/'
        creatinine='_creatinine'
    
    path='../../Data/Results/'+path_to_add  
    imputation_method='knn'
    constant_end_of_path=str(number_of_patients)+'_patients_test_size_'+str(test_size)+imputation_method


    with open(path+'AUC_results/auc_test_all_with_'+constant_end_of_path,'rb') as f:
        AUC_test=pickle.load(f)

    with open(path+'Probabilities/probabilities_with_'+constant_end_of_path,'rb') as f:
        Probabilities=pickle.load(f)

            #print(AUC_test['target_AKI'])

    models_list = []
    seeds_list = []
    target_list = []
    AUC_list = []
    PR_list = []
    Baseline_list = []


    for target in targets_to_test:
        for seed in seeds:
            for model_name in models_to_test:
                models_list.append(model_name)
                seeds_list.append(seed)
                target_list.append(target)
                AUC_list.append(AUC_test[target][model_name][seed])

                true_y = Probabilities[target][model_name][seed][target].values
                preds = Probabilities[target][model_name][seed]['proba_'+target].values
                average_precision = average_precision_score(true_y,preds)
                PR_list.append(average_precision)
                Baseline_list.append(true_y.mean())

    df =pd.DataFrame([models_list, seeds_list,target_list, AUC_list,PR_list,Baseline_list]).T
    df.columns = ['models', 'seeds','target', 'AUC','PR','Baseline']

    AUC_analysis=create_mean_CI(df,confidence,text='AUC')
    PR_analysis= create_mean_CI(df,confidence,text='PR')
#Baseline_analysis= create_mean_CI(df,confidence,text='Baseline')
    if save_AUC:
        AUC_analysis.to_csv(path+'AUC_results/AUC_ROC_all_with_'+constant_end_of_path+'.csv')

    df = AUC_analysis.merge(PR_analysis,on=['target','models'])
#df = AUC_both.merge(Baseline_analysis,on=['target','models'])

    if save_AUC:
        df.to_csv(path+'AUC_results/AUC_ROC_all_with_'+constant_end_of_path+'.csv')

    Targets_analysis = df.groupby('target').agg({'AUC_mean':'max','PR_mean':'max',}).sort_values(by=['PR_mean'],ascending=False).reset_index()

    #argets_analysis['Ratio_PR']=Targets_analysis['PR_mean']/Targets_analysis['Baseline_mean']
    if all_CI:
        return df
    if not all_CI:
        return Targets_analysis

#Regression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

def get_or_save_MSE_csv(
                       targets_to_test,
                       models_to_test,
                       seeds,
                       number_of_patients,
                       test_size=0.25,
                       save_AUC=False,
                       confidence=0.95,
                       is_Creatinine=False,
                       all_CI=False):
    
    
    
    
    path_to_add='Hemoglobin/'
    creatinine='_hemoglobin'
    continuous='_continuous'
    if is_Creatinine:
        path_to_add='Creatinine/'
        creatinine='_creatinine'
    
    path='../../Data/Results/'+path_to_add
    #print(path)
    imputation_method='knn'
    constant_end_of_path=str(number_of_patients)+'_patients_test_size_'+str(test_size)+imputation_method+continuous


    with open(path+'AUC_results/auc_test_all_with_'+constant_end_of_path,'rb') as f:
        AUC_test=pickle.load(f)
    
    with open(path+'Probabilities/probabilities_with_'+constant_end_of_path,'rb') as f:
        Probabilities=pickle.load(f)


    models_list = []
    seeds_list = []
    target_list = []
    MAE_list = []
    MSE_list = []
    R2_list = []



    for target in targets_to_test:
        for seed in seeds:
            for model_name in models_to_test:
                models_list.append(model_name)
                seeds_list.append(seed)
                target_list.append(target)
            
                
                true_y = Probabilities[target][model_name][seed][target].values
                preds = Probabilities[target][model_name][seed]['proba_'+target].values
                mae = mean_absolute_error(true_y,preds)
                mse = mean_squared_error(true_y,preds)
                r2 = r2_score(true_y,preds)
                MAE_list.append(mae)
                MSE_list.append(mse)
                R2_list.append(r2)

                

    df =pd.DataFrame([models_list, seeds_list,target_list, MAE_list,MSE_list,R2_list]).T
    df.columns = ['models', 'seeds','target', 'MAE','MSE','R2']

    MAE_analysis=create_mean_CI(df,confidence,text='MAE')
    MSE_analysis= create_mean_CI(df,confidence,text='MSE')
    R2_analysis= create_mean_CI(df,confidence,text='R2')

    MAE_both = MSE_analysis.merge(MAE_analysis,on=['target','models'])
    df = MAE_both.merge(R2_analysis,on=['target','models'])

    df.sort_values(by=['MAE_mean'],ascending=True,inplace=True)
        
    if save_AUC:
        df.to_csv(path+'AUC_results/AUC_ROC_all_with_'+constant_end_of_path+'.csv')

    Targets_analysis = df.groupby('target').agg({'MAE_mean':'max','MSE_mean':'max','R2_mean':'max'}).sort_values(by=['MAE_mean'],ascending=True).reset_index()

    #argets_analysis['Ratio_PR']=Targets_analysis['PR_mean']/Targets_analysis['Baseline_mean']
    if all_CI:
        return df
        if not all_CI:
            return Targets_analysis

    #AUC_analysis,PR_analysis


def create_mean_CI(df,confidence,text='AUC'):

    AUC_mean = df.groupby(('models','target')).agg(text).apply(lambda x :round(np.mean(x),3))
    
    std = df.groupby(('models','target')).agg(text).apply(lambda x :round(np.std(x),3))
    
    CI = df.groupby(('models','target')).agg(text).apply(lambda x: mean_confidence_interval(x,confidence=confidence))
    
    
    AUC_analysis = pd.DataFrame(AUC_mean).reset_index()
    AUC_analysis.rename(columns={text:text+'_mean'},inplace=True)
    
    AUC_analysis = AUC_analysis.merge(pd.DataFrame(CI).reset_index(),how='left',on=['models','target'])
    AUC_analysis.rename(columns={text:text+'_CI_'+str(confidence)},inplace=True)
    AUC_analysis = AUC_analysis.merge(pd.DataFrame(std).reset_index(),how='left',on=['models','target'])
    AUC_analysis.rename(columns={text:text+'_std'},inplace=True)
    
    AUC_analysis = AUC_analysis[[ 'target','models', text+'_mean', text+'_CI_0.95', text+'_std']]
    AUC_analysis.sort_values(by=['target',text+'_mean'],ascending=False,inplace=True)
    return AUC_analysis


