# -------------------------------------------------------------

# DESCRIPTION
# This is a helper file that help us run models and boostrapping

# This file define the hyperparameters of each models
# It creates a fucntion to fetch exisiting results to update them and not erase everytime
# There are two main functions in this file:
# "run" is function to run a model for a given target and seed and output the AUC on test, the feature importance and the probabilities
#run_models is the main function, that calls "run" for different combination of target,models, and seeds. it also feature and option to run the code in parallel. Especially useful in Google cloud. One should pay attention to concurrency issues though when the parameters of models are already defined in parallel (eg: RF and Log have n_jobs=-1).

# INPUT
# all the parameters that we want to run in the bootstrap

# OUPUT
# Importance
# Probabilities
# AUC


import pytz
central= pytz.timezone("US/Central")

# Basic packages
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

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

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


from sklearn.metrics import classification_report,confusion_matrix
import itertools

import time
from threading import Thread

run_OCT=False

if run_OCT==True:
    from julia import Julia
    Julia(runtime='/Applications/Julia-0.7.app/Contents/Resources/julia/bin/julia',compiled_modules=False)

    from interpretableai import iai
    from julia import Distributed
    Distributed.addprocs(5)
    models1 = {"RFC":RandomForestClassifier, "XGB":GradientBoostingClassifier,"Log":LogisticRegression,"CART":tree.DecisionTreeClassifier,"OCT":iai.OptimalTreeClassifier}
    models2 = {"RFC":RandomForestRegressor, "XGB":GradientBoostingRegressor,"Lin":LinearRegression,"ORT":iai.OptimalTreeRegressor}






path_to_load='../../Data/Generated_csv/'
imputation_method='fillna'




# # 3- Define functions and parameters


# Define parameters of models

models1 = {"RFC":RandomForestClassifier, "XGB":GradientBoostingClassifier,"Log":LogisticRegression,"CART":tree.DecisionTreeClassifier}


models2 = {"RFC":RandomForestRegressor, "XGB":GradientBoostingRegressor,"Lin":LinearRegression}


# change the parameters to fit grid search
params = {
    "RFC":{'max_features': None,
        'bootstrap': True,
        'min_samples_leaf': 30,
        'max_depth' :30,
        'n_estimators': 400,
        'n_jobs': -1,
        'random_state':42},
    
    "XGB":{'criterion': 'friedman_mse', 'subsample': 1, 'max_features': None, 'min_samples_leaf': 30, 'learning_rate': 0.075, 'random_state': 42, 'n_estimators': 50, 'max_depth': 5},

    "Log":{'penalty':'l1',
         'max_iter':100,
         'C': 0.05,
         'n_jobs': -1,
         'random_state': 42},

    "Lin":{},

    "CART":{},

# do OCT grid search and then decide the parameters
    #change parameter
    #"OCT":{'max_depth':12,
    #    #'cp': 0.002, #crea and filtered
    #    #'cp': 0.008, # hemo
    #    'cp': 0.0009570757701315991, # hemo
    #    'minbucket':20,
    #    'ls_num_tree_restarts':400,
    #    'random_seed':1,
    #    'criterion':"gini"},

    "OCT":{'max_depth':9,
        'cp': 0.0006,
        'minbucket':30,
        'ls_num_tree_restarts':300,
        'random_seed':1,
        'criterion':"gini"},

    "ORT":{'max_depth':10,
        'cp': 0.002, #crea
        #'cp': 0.0009570757701315991, # hemo
        'minbucket':20,
        'ls_num_tree_restarts':400,
        'random_seed':1,
        'criterion':"mse"}
         }


path='../../Data/Results/'





def initialize_or_fetch_dict(path,header,seeds,targets_to_test,models_to_test,constant_end_of_path):
    
    try:
        with open(path+header+constant_end_of_path,'rb') as f:
            dict_to_take=pickle.load(f)
    except FileNotFoundError:
        dict_to_take = {}
    for target in targets_to_test:
        # initialize dict
        try:
            dict_to_take[target]
            for model_name in models_to_test:
                try:
                    dict_to_take[target][model_name]
                except:
                    dict_to_take[target][model_name]={}
        except:
            dict_to_take[target]={}
            for model_name in models_to_test:
                dict_to_take[target][model_name]={}
    return dict_to_take




# In[71]:


def run_models(
    number_of_patients,
    test_size,
               #models=models,
               #params=params,
    X,
    y,
    seeds,
    targets_to_test,
    models_to_test,
    run_parralel=False,
    save_csv=False,
    save_probabilities=True,
    save_models=False,
    save_importance = True,
    save_auc=True,
    creatinine=False,
    imputation_method='mean',
    categoric_features=['ethnicity'],
    is_continuous=False
               ):
    oct_mode=False
    if 'OCT' in models_to_test or 'ORT' in models_to_test:
        oct_mode=True
    
    continuous=''
    models=models1
    if is_continuous:
        models=models2
        continuous='_continuous'

    if ['Baseline'] == models_to_test:
        save_importance=False
    
    if not oct_mode:
        X =pd.get_dummies(X,columns=categoric_features)
    
    if oct_mode:
        X[categoric_features] = X[categoric_features].astype('category')


    path_to_add='Hemoglobin/'
    if creatinine:
        path_to_add='Creatinine/'
    path='../../Data/Results/'+path_to_add

    
    
    constant_end_of_path=str(number_of_patients)+'_patients_test_size_'+str(test_size)+imputation_method+continuous
    
    # if file already created just load them and update
    
    AUC_test_dict = initialize_or_fetch_dict(path,'AUC_results/auc_test_all_with_',seeds,targets_to_test,models_to_test,constant_end_of_path)

    Importances = initialize_or_fetch_dict(path,'Features/feature_importance_all_with_',seeds,targets_to_test,models_to_test,constant_end_of_path)

    Probabilities = initialize_or_fetch_dict(path,'Probabilities/probabilities_with_',seeds,targets_to_test,models_to_test,constant_end_of_path)

    try:
        Importance_all = pd.read_csv(path+'Features/feature_importance_all_with_'+constant_end_of_path+'.csv')
    except FileNotFoundError:
        Importance_all = pd.DataFrame(columns=['Feature','Importance','Model','seed','target'])
    
    if run_parralel:
        threads = []
        
        for target in targets_to_test:
            for model_name in models_to_test:
                # bug when RFC runs in parallel
                if model_name=='RFC':
                    for seed in seeds:
                        run(
                            model_name,
                            seed,
                            target,
                            number_of_patients,
                            test_size,
                            models, 
                            params,
                            X,
                            y,
                            AUC_test_dict,
                            Importances,
                            Probabilities,
                            save_csv,
                            save_probabilities,
                            save_models,
                            imputation_method,
                            creatinine,
                            is_continuous
                               )
                    
                if model_name!='RFC':
                    for seed in seeds:
                        # We start one thread per combination of seed/model/target
                        process = Thread(
                            target=run,
                            args=[
                                model_name,
                                seed,
                                target,
                                number_of_patients,
                                test_size,
                                models,
                                params,
                                X,
                                y,
                                AUC_test_dict,
                                Importances,
                                Probabilities,
                                save_csv,
                                save_probabilities,
                                save_models,
                                imputation_method,
                                creatinine,
                                is_continuous
                            ])
                        process.start()
                        threads.append(process)
        # We now pause execution on the main thread by 'joining' all of our started threads.
        # This ensures that each has finished processing.
        for process in threads:
            process.join()
            
    # when not in parrallel
    if not run_parralel:
        for target in targets_to_test:
            for model_name in models_to_test:
                for seed in seeds:
                    run(
                        model_name,
                        seed,
                        target,
                        number_of_patients,
                        test_size,
                        models,
                        params,
                        X,
                        y,
                        AUC_test_dict,
                        Importances,
                        Probabilities,
                        save_csv,
                        save_probabilities,
                        save_models,
                        imputation_method,
                        creatinine,
                        is_continuous
                           )

    if save_importance:
        for target in targets_to_test:
            for model_name in models_to_test:
                for seed in seeds:
                    Importance_all = pd.concat((Importance_all,Importances[target][model_name][seed]))
    
        Importance_all.to_csv(path+'Features/feature_importance_all_with_'+constant_end_of_path+'.csv')
        f = open(path+'Features/feature_importance_all_with_'+constant_end_of_path,"wb")
        pickle.dump(Importances,f)

    if save_auc:
        f = open(path+'AUC_results/auc_test_all_with_'+constant_end_of_path,"wb")
        pickle.dump(AUC_test_dict,f)
        f.close()

    if save_probabilities:
        f = open(path+'Probabilities/probabilities_with_'+constant_end_of_path,"wb")
        pickle.dump(Probabilities,f)
        f.close()
    return AUC_test_dict




# In[72]:

def run(
    model_name,
    seed,
    target,
    number_of_patients,
    test_size,
    models, 
    params,
    X,
    y,
    AUC_test_dict,
    Importances,
    Probabilities,
    save_csv=False,
    save_probabilities=True,
    save_models=False,
    imputation_method='mean',
    is_creatinine=False,
    is_continuous=False
       ):
    
    
    start=time.time()
    time_now = datetime.now(central)
    print("Time when start: ",time_now.strftime("%H:%M:%S"))

    continuous=True
    if target=='target_bin':
        continuous=False
    
    path_to_add='Hemoglobin/'
    if is_creatinine:
        path_to_add='Creatinine/'


    path_to_save='../../Data/Generated_csv/'+path_to_add
    
    path_to_save_models='../../Data/Results/'+path_to_add+'Models/'
    
    
    #run only on a sample of the patients
    X_sample = X.iloc[:number_of_patients]
    y_sample = y.iloc[:number_of_patients]
    
    print(seed)

    #sample
    if not is_continuous:
        train_X, test_X, train_y, test_y = train_test_split(X_sample, y_sample,stratify=y_sample[target], test_size=test_size, random_state=seed)

    if is_continuous:
        train_X, test_X, train_y, test_y = train_test_split(X_sample, y_sample, test_size=test_size, random_state=seed)

            
    # function to test a model trained on all the data but tested on a filtered testing set
    filter_after=False
    if filter_after:
        test_X = test_X.loc[test_X.Hb_value_initial<=10]
        test_y= test_y.loc[test_y.index.isin(test_X.index)]


    cols = train_X.columns
    if model_name=='Log' or model_name=='Lin':
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X )

    if save_csv:
        #save the csv for reproductibility and OCT
        train_X.to_csv(path_to_save+'train_and_test/'+imputation_method+'_train_X_seed_'+str(seed)+'.csv')
        test_X.to_csv(path_to_save+'train_and_test/'+imputation_method+'_test_X_seed_'+str(seed)+'.csv')
        train_y.to_csv(path_to_save+'train_and_test/'+imputation_method+'_train_Y_seed_'+str(seed)+'.csv')
        test_y.to_csv(path_to_save+'train_and_test/'+imputation_method+'_test_Y_seed_'+str(seed)+'.csv')


    
# Uncomment this code to do a grid search within the boostrapping
#if model_name=='OCT':
#    crit='gini'
#    print('starting grid search')
#
#    grid = iai.GridSearch(
#                       iai.OptimalTreeClassifier(
#                                                 random_seed=1,
#                                                 ),
#                       max_depth=range(5, 7), #do range
#                       criterion=crit,
#                       ls_num_tree_restarts=300
#                       )
#    grid.fit(train_X, train_y[target])
#    model = grid.get_learner()
#    print(grid.score(test_X, test_y[target], criterion='auc'))
#    print(grid.get_best_params())

#elif model_name=='ORT':
#     crit='mse'
#     print('starting grid search ORT')
#
#     grid = iai.GridSearch(
#                           iai.OptimalTreeRegressor(
#                                                    random_seed=1,
#                                                    ),
#                           max_depth=range(3, 10), #do range
#                           criterion=crit,
#                           ls_num_tree_restarts=300)
#     grid.fit(train_X, train_y[target])
#     model = grid.get_learner()
#     print(grid.score(test_X, test_y[target], criterion='mse'))
#     print(grid.get_best_params())

    
    elif model_name!='Baseline':
        model=models[model_name](**params[model_name])
        model.fit(train_X, train_y[target])



    # uncomment this code if you want to load and already trained tree
    
    #if model_name=='OCT':
    #file_path=path_to_save_models+model_name+'_'+target+'_'+str(number_of_patients) +'_patients_test_size_'+str(test_size)+'mean'+'_seed'+str(1)
    #model=iai.read_json(file_path)
        
    if not is_continuous:

        if model_name=='OCT':
            preds = model.predict_proba(test_X)['1']
        elif model_name=='Baseline':
            preds = test_y[target+'_baseline'].values
        else:
            preds = model.predict_proba(test_X)[:,1]
        AUC_test= roc_auc_score(test_y[target], preds)
        if not is_creatinine:
            test_X.reset_index(inplace=True)
            test_y.reset_index(inplace=True)
            # test the performances on a filtered dataset
            #subset_list = test_X.loc[test_X.Hb_value_initial<10].index
            #suberror = roc_auc_score(test_y.loc[test_y.index.isin(subset_list)][target], preds[subset_list])
            #print("Normal :",AUC_test)
            #print("Sub :",suberror)
            # if want to ouptut error on subgroup of patients
            #AUC_test=suberror


    if is_continuous:
        if model_name=='ORT':
            preds = model.predict(test_X)
        elif model_name=='Baseline':
            preds = test_y[target+'_baseline'].values
        else:
            preds = model.predict(test_X)
        AUC_test= mean_absolute_error(test_y[target], preds)
        if not is_creatinine:
            test_X.reset_index(inplace=True)
            test_y.reset_index(inplace=True)
            #subset_list = test_X.loc[test_X.Hb_value_initial<8].index
            #suberror = mean_absolute_error(test_y.loc[test_y.index.isin(subset_list)][target], preds[subset_list])
            #print("Normal :",AUC_test)
            #print("Sub :",suberror)
            # if want to ouptut error on subgroup of patients
            #AUC_test=suberror

    
    AUC_test_dict[target][model_name][seed]=AUC_test

    if save_probabilities:
        prob=pd.DataFrame()
        prob[target]=test_y[target]
        #prob['proba_'+target]=preds.reset_index()
        if model_name=='OCT':
            prob['proba_'+target]=preds.values
        else:
            prob['proba_'+target]=preds
        Probabilities[target][model_name][seed]=prob
    if model_name=='Baseline':
        return AUC_test
        
    if save_models:
        file_path=path_to_save_models+model_name+'_'+target+'_'+str(number_of_patients) +'_patients_test_size_'+str(test_size)+imputation_method+'_seed'+str(seed)
        if model_name=='OCT' or model_name=='ORT':
            model.write_json(file_path)
            model.write_html(file_path+'.html')
        else:
            f = open(file_path,"wb")
            pickle.dump(model,f)
            f.close()

    #add feature importance

    if model_name=='OCT' or model_name=='ORT':
        Importance =model.variable_importance()
    else:
        if model_name=='Log':
            features= model.coef_[0]
        elif model_name=='Lin':
            features= model.coef_

        else:
            features=model.feature_importances_
        Importance = pd.DataFrame(features,columns=['Importance'])
        Importance['Feature']=list(cols)
    Importance['Model']=model_name
    Importance['seed']=seed
    Importance['target']=target
    Importances[target][model_name][seed] = Importance

    print("End of one thread!")
    stop =time.time()
    print("took %.2f seconds"% ((stop - start)))
    if save_probabilities:
        return AUC_test,Importance,prob
    else:
        return AUC_test


