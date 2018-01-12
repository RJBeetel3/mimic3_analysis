import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

from scipy.stats import ks_2samp
import scipy.stats as scats


from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from heapq import nlargest
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import cPickle as pickle

#import datetime as datetime
#from dateutil.relativedelta import relativedelta
#from sklearn.preprocessing import OneHotEncoder
#import psycopg2
#from scipy.stats import boxcox
#import visuals as vs
#import math





def import_features():
    features_dir = '../data/features/'
   
    # look into re-doing this using *.csv   
    monkey = os.listdir(features_dir)
    feature_files = [x for x in monkey if '.csv' in x]
    
    features_dict = {}
    for name in feature_files:
        #print(name)
        features_dict[name.split('.')[0]] = pd.DataFrame.from_csv(features_dir + name)
    outcomes = pd.DataFrame.from_csv(features_dir + 'outcomes.csv')

    scores = [x for x in features_dict.keys() if 'Scores' in x]
    features = [x for x in features_dict.keys() if 'Scores' not in x]
    scores.sort()
    features.sort()
    
    

    for i in range(len(scores)):
        if i == 0:
            all_scores = features_dict[scores[i]]
        else:
            all_scores = all_scores.append(features_dict[scores[i]])


    all_scores = all_scores.sort_values(by = 'p_values', ascending = True)
    top_scores = list(all_scores.head(20).index)
    
    #print "scores"
    #print(all_scores)
   
    
    

    first = True
    all_data = outcomes
    for frame in features:
        for col in features_dict[frame].columns:
            if col in top_scores:
            
                feat = features_dict[frame][col] 
                all_data = all_data.merge(pd.DataFrame(feat), left_index = True, 
                                         right_index = True, how = 'inner', sort = True)
            
                #display(feat.name)
                #display(all_data.shape)

    all_data.rename(index=str, columns={"Pneumonia (except that caused by tuberculosis or sexually transmitted disease)": "Pneumonia",
                                         "Respiratory failure; insufficiency; arrest (adult)": "Respiratory Failure"}, 
                     inplace = True)
    
    return all_data
    
    
def block_features(all_data):
    
    # SPLITTING FEATURES INTO TWO SETS, THOSE RELATED TO DIAGNOSES AND THOSE
    # RELATED TO MEASUREMENTS TAKEN
    
    diagnoses_list = ['hospital_expire_flag', 'GCS Total_15', 'GCS Total_3', 'GCS Total_6', 
                        'GCS Total_4', 'Respiratory Failure','Shock', 
                        'Septicemia (except in labor)', 'Acute and unspecified renal failure',
                        'Other liver diseases', 'Fluid and electrolyte disorders','Pneumonia',
                            'Acute cerebrovascular disease'
                    ]
    
    meas_list = ['hospital_expire_flag', 'Creat_abnflag','icu_stay_Q3','first_careunit_CSRU',
                              'admission_type_ELECTIVE', 'icu_stay_Q0','age_Q3',
                              'first_careunit_MICU'
                ]
  
    diagnoses_features = all_data[diagnoses_list]
    meas_features = all_data[meas_list]    
    
    return diagnoses_features , meas_features
    
def plot_survival_rates(features, save_flag):
    
    # PLOT SURVIVAL RATES FOR PATIENTS IN WHICH EACH CONDITION IS TRUE
    # FOR EACH FEATURE, PLOT SURVIVAL RATES FOR PATIENT WITH CONDITION = TRUE
    
    live_dead_dict = {}
    

    for col in features.columns[1:]: 
        # ALL DATA FOR WHICH 
        positives = features[features[col] == 1] 
        dead = positives.hospital_expire_flag.sum()
        total = positives.hospital_expire_flag.count()
        dead_percent = 100.0*dead/total
        live_percent = 100.0*(total-dead)/total
        live_dead_dict[col] = [dead_percent, live_percent]
    
    live_dead_df =pd.DataFrame.from_dict(live_dead_dict)
    live_dead_df.index = [['Non-Survivors', 'Survivors']]
    
    
    
    live_dead_df.transpose().plot.bar(stacked = True, figsize = (10,7), edgecolor = 'black', linewidth = 3, 
                                    alpha = 0.5, title = "Percent of Survivors and Non-Survivors for \n ICU Stays where Each Condition is True")
    
    plt.subplots_adjust(left = 0.1, bottom = 0.4, 
                    right = 0.9, top = 0.9) 
    
    plt.legend(loc="upper left", bbox_to_anchor=(0.75,0.75),fontsize=12)
    plt.xticks(rotation = 60, ha = 'right')

    if save_flag:
        save_file_name = "../figures/survival_rates.png"
        print "saving {}".format(save_file_name)
        plt.savefig(save_file_name)
    
    plt.close()

    
    
def plot_true_rates(features, save_flag):
    
    # PLOT THE RATES OF PATIENTS IN WHICH THE VALUE OF A GIVEN FEATURE IS TRUE
    # FOR SURVIVOR AND NON SURVIVOR GROUPS. 
    dead_positive_dict = {}
    non_survivors = features[features.hospital_expire_flag == 1]

    # CALCULATION OF THE PERCENT OF NON-SURVIVORS FOR WHICH EACH CONDITION IS TRUE AND FALSE
    for col in non_survivors.columns[1:]:
        # NUMBER OF NON-SURVIVORS FOR WHICH CONDITION IS TRUE
        dead_positive = non_survivors[col].sum()
        total = non_survivors[col].count()
        # PERCENTAGE OF NON-SURVIVORS FOR WHICH CONDITION IS FALSE AND TRUE
        positive_percent = 100.0*dead_positive/total
        negative_percent = 100.0*(total-dead_positive)/total
        dead_positive_dict[col] = [positive_percent, negative_percent]
    
    dead_positive_df =pd.DataFrame.from_dict(dead_positive_dict)
    dead_positive_df.index = [['Percent of Non_Survivors where Condition = True', 'Live']]
    #display(dead_positive_df)
    '''
    live_dead_df.transpose().plot.bar(stacked = True, figsize = (13,6), edgecolor = 'black', linewidth = 3, 
                                    alpha = 0.5, title = "Percent of Non_Survivor Samples with Positive Label")
    '''

    live_positive_dict = {}
    survivors = features[features.hospital_expire_flag == 0]

    # CALCULATION OF THE PERCENT OF SURVIVORS FOR WHICH EACH CONDITION IS TRUE AND FALSE
    for col in survivors.columns[1:]:
        # NUMBER OF SURVIVORS FOR WHICH CONDITION IS TRUE
        live_positive = survivors[col].sum()
        total = survivors[col].count()
        # PERCENT OF SURVIVORS FOR WHICH CONDITION IS TRUE AND FALSE
        positive_percent = 100.0*live_positive/total
        negative_percent = 100.0*(total-live_positive)/total
        live_positive_dict[col] = [positive_percent, negative_percent]
    
    live_positive_df=pd.DataFrame.from_dict(live_positive_dict)
    live_positive_df.index = [['Percent of Survivors for which Condition = True', 'Live2']]
    #display(live_positive_df)

    positive_df = dead_positive_df.append(live_positive_df)
    #display(positive_df[list(diagnoses_features.columns[1:])])
    # PLOTTING PERCENTAGES OF SURVIVORS AND NON-SURVIVORS FOR WHCIH CONDITION IS TRUE. 
    # PLOTTING DIAGNOSES FEATURES IN ONE PLOT
    
    ax1 = positive_df.transpose()[[
            'Percent of Non_Survivors where Condition = True',
            'Percent of Survivors for which Condition = True']].plot.bar(
            stacked = False, figsize = (10,7), edgecolor = 'black', 
            linewidth = 3, alpha = 0.5, 
            title = "Percent of Survivors and Non-Survivors for which Condition is True")
            
    plt.xticks(rotation = 60, ha = 'right')
    ax1.set_ylim([0,100]) 
    
    ax1.legend(loc="upper left", bbox_to_anchor=(0.5,0.9),fontsize=12)
    plt.subplots_adjust(left = 0.1, bottom = 0.4, right = 0.9, top = 0.9) 
   
    
   
    # PLOTTING REST OF FEATURES IN ANOTHER PLOT
   
    if save_flag:
        save_file_name1 =  "../figures/condition_true.png"
       
        print "saving {}".format(save_file_name1)
        plt.savefig(save_file_name1)
    
    plt.close()




def create_benchmarks(all_data):

    # CREATING BASELINE SCORES USING 20 INPUT FEATURES AND 
    # CLASSIFIERS WITH DEFAULT SETTINGS

    SVC = svm.SVC()
    Kneighb = KNeighborsClassifier()
    LSVC = svm.LinearSVC()
    MLP = MLPClassifier()
    Tree = DecisionTreeClassifier() 

    default_clfs = [SVC, Kneighb, LSVC, MLP, Tree]

    X_best = all_data[all_data.columns[1:]]
    y = all_data['hospital_expire_flag']

    X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.20, random_state = 42)

    first = True

    for clf in default_clfs:        
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        scores_report = metrics.classification_report(y_true, y_pred)
        #print type(clf).__name__
        #print scores_report
            
        
        # CONVERT SCORES_REPORT TO DATAFRAME
        scores = scores_report.split()
        #display(scores)
        cols = scores[:4]
        #cols = ['default ' + x for x in cols]
   
        ind = ['Survivors', 'Non-Survivors', 'Avg/Total']
        # CLEAN THIS UP
        dat = [[float(x) for x in scores[5:9]], [float(x) for x in scores[10:14]], 
                                                [float(x) for x in scores[17:22]]]
    
        key = type(clf).__name__
        #print key
        arrays = [[key, key, key], ind]
        tuples = list(zip(*arrays))
        mindex = pd.MultiIndex.from_tuples(tuples, names=['Classifier', 'Classes'])
        if first:
            def_scores = pd.DataFrame(dat, columns = cols, index = mindex)
            first = False
        else: 
            def_scores = def_scores.append(pd.DataFrame(dat, columns = cols, index = mindex))
        

    def_scores.name = "Default Classifier Scores"
    #def_scores.sort_index()
    return def_scores 
    
    
    
def plot_benchmarks(def_scores, save_flag):

    monkey = def_scores.copy()
    classers = list(monkey.index.levels[0])
    metric_dict = {}

    monkey['f1 metric'] = 0.00
    monkey['recall metric'] = 0.00

    for clfr in classers:
        f1survs = monkey['f1-score'][clfr, 'Survivors']
        f1deads = monkey['f1-score'][clfr, 'Non-Survivors']
        f1met = (f1survs+f1deads)-abs(f1survs-f1deads)
        recall_survs = monkey['recall'][clfr, 'Survivors']
        recall_deads = monkey['recall'][clfr, 'Non-Survivors']
        recallmet = (recall_survs+recall_deads)-abs(recall_survs-recall_deads)
        #display(recallmet)
        metric_dict[clfr] = [f1met, recallmet]
        monkey['f1 metric'][clfr, 'avg/total'] = f1met
        monkey['recall metric'][clfr, 'avg/total'] = recallmet

    #display(monkey)
    bm_metric_df = pd.DataFrame.from_dict(metric_dict)
    bm_metric_df.index = ['F1 Metric', 'Recall Metric']
    bm_metric_df = bm_metric_df.transpose()
    print(bm_metric_df)





    #ax1 = bl_metric_df['F1 Metric'].plot.bar(legend = True, figsize = (13,8), rot = 60)
    #ax2 = bl_metric_df['Recall Metric'].plot.bar(secondary_y=True, label = 'Recall Metric', legend = True, 
    #                                         rot = 60)
    ax1 = bm_metric_df.plot.bar(legend = True, figsize = (13,8), rot = 60)
    ax1.set_ylim(0,1.5)
    #ax2.set_ylim(0,1.5)
    plt.subplots_adjust(left = 0.1, bottom = 0.25, 
                    right = 0.9, top = 0.9) 
    plt.title('Benchmark Optimization Metric Scores for Classifiers with Default Parameters\n Trained and Tested on 20 Features')
    
    if save_flag:
        save_file_name1 =  "../figures/classifier_benchmarks.png"
   
        print "saving {}".format(save_file_name1)
        plt.savefig(save_file_name1)
    
    plt.close()
    
    #plt.show()

def optimize_SVC(all_data, optimized_clfs):

    # THE SUPPORT VECTOR CLASSIFIER IS OPTIMZED TO TO F1-SCORE AND RECALL OVER THE GIVEN PARAMETER SPACE
    # AND OVER THE NUMBER OF FEATURES USING THE TOP 20 FEATURES RANKED ACCORDING TO CHI2 SCORE
    # RECALL FOR MORTALITY PREDICTION IS IMPORTANT IN THAT WE DON'T WANT FALSE NEGATIVES. 
    # THE SUPPORT VECTOR CLASSIFIER TAKES A VERY LONG TIME TO OPTIMIZE OVER THE GIVEN PARAMETER SPACE SO 
    # THAT IS BEING DONE INDIVIDUALLY HERE. THE REMAINING CLASSIFIERS ARE OPTIMIZED BELOW

    

    #LinearSVC CLASSIFIER PARAMETERES
    SVC_params = {'C':[0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1],  
                   'class_weight': [{1:3, 0:1}, {1:4, 0:1}, {1:5, 0:1}],
                   'kernel': ['rbf', 'sigmoid', 'poly',], 
                   'degree': [2,3,4], 
                   'decision_function_shape': ['ovr','ovo']
                                
                  }


    # CLASSIFIER SCORES
    scores = ['f1', 'recall'] #'accuracy', , 'precision']

    # DICTIONARY OF CLASSIFIERS AND CORRESPONDING PARAMETERS 
    SVC = svm.SVC(random_state = 42)
    SVC_stuff = [SVC, SVC_params]

    # CREATED ARCHITECTURE FOR OPTIMIZING MULTIPLE CLASSIFIERS IF SO DESIRED
    # BUT IN OUR CASE WE'RE ONLY OPTIMIZING ONE. 
    
    classifiers = {
                   'SVC' : SVC_stuff
                  }
               


    for key in classifiers.keys():
    
        recall_mortality = -100
        f1_mortality = -100
        optimized_params = {}
        optimized = {}
        optimized_scores = ""
        recall_scores = {}
        f1_scores = {}

        print "Evaluating {}".format(key)
        for num_feats in range(8, all_data.shape[1]):
            print "{} features".format(num_feats)
            # FEATURES AND TARGETS
            X_best = all_data[all_data.columns[1:num_feats]]
            y = all_data['hospital_expire_flag']

            X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.30, random_state = 42)
        
        
            # MAY NOT BE NECESSARY THE WAY THE CODE IS CURRENTLY BROKEN OUT
            for score in scores:
                #print"# Optimizing parameters for {} classifier to {} score using {} features".format(key, score, num_feats)

         
                clf = GridSearchCV(classifiers[key][0], classifiers[key][1], cv = 5, 
                               scoring = score)

                clf.fit(X_train, y_train)
    
                y_true, y_pred = y_test, clf.predict(X_test)
                scores_report = metrics.classification_report(y_true, y_pred)
            
            
                #print(clf.best_params_)
                #print(scores_report)
         
                if score == 'recall':
                    surv, mort = metrics.recall_score(y_true, y_pred, average = None)
                    # a metric that maximizes both variables and minimizes the difference between them
                    recall_metric = mort + surv - (abs(mort-surv))
                    recall_scores[str(num_feats)] = [surv, mort, recall_metric]
                    if (recall_metric > recall_mortality):
                        recall_mortality = recall_metric #math.sqrt(mort**2 + surv**2)
                        optimized = clf.best_params_
                        optimized['features'] = num_feats
                        optimized_scores = scores_report
                        recall_str = key + '_recall'
                        if recall_str not in optimized_clfs:
                            optimized_clfs[recall_str] = {}
                        optimized_clfs[recall_str]['CLF'] = clf.best_estimator_
                        optimized_clfs[recall_str]['PARAMS'] = optimized
                        optimized_clfs[recall_str]['SCORES'] = optimized_scores
                   
         
                elif score == 'f1':
            
                    surv_f1, mort_f1 = metrics.f1_score(y_true, y_pred, average = None)
                    #a metric that maximizes both variables and minimizes the difference between them
                    f1_metric = surv_f1 + mort_f1 - (abs(surv_f1-mort_f1))
                    f1_scores[str(num_feats)] = [surv_f1, mort_f1, f1_metric]
                    if (f1_metric > f1_mortality):
                        f1_mortality = f1_metric #math.sqrt(mort**2 + surv**2)
                        #recall_delta = abs(mort-surv)
                        f1_optimized = clf.best_params_
                        f1_optimized['features'] = num_feats
                        f1_optimized_scores = scores_report
                        f1_str = key + '_f1'
                        if f1_str not in optimized_clfs.keys():
                            optimized_clfs[f1_str] = {}
                        optimized_clfs[f1_str]['CLF'] = clf.best_estimator_
                        optimized_clfs[f1_str]['PARAMS'] = f1_optimized
                        optimized_clfs[f1_str]['SCORES'] = f1_optimized_scores
                    

        print "ANALYSIS COMPLETE"
        recalls = pd.DataFrame.from_dict(recall_scores)
        recalls = recalls.transpose()
        recalls['features'] = recalls.index
        recalls['features'] = recalls['features'].apply(lambda x: int(x))
        recalls.sort_values(by = 'features', inplace = True)
        recalls.columns = ['Survivors', 'Non-Survivors', 'Metric', 'Features']
        #recalls[['Survivors', 'Non-Survivors', 'Metric']].plot(title = 'Recall Scores', 
        #                                                       use_index = True, figsize = (16,6))
        #plt.show()
        print "Optimized Parameters for the {} Classifier for recall are".format(key)
        print(optimized)
        print "optimized Scores are"
        print(optimized_scores)
       
    
        f1s = pd.DataFrame.from_dict(f1_scores)
        f1s = f1s.transpose()
        f1s['features'] = f1s.index
        f1s['features'] = f1s['features'].apply(lambda x: int(x))
        f1s.sort_values(by = 'features', inplace = True)
        f1s.columns = ['Survivors', 'Non-Survivors', 'Metric', 'Features']
        #f1s[['Survivors', 'Non-Survivors', 'Metric']].plot(title = 'F1 Scores', use_index = True, figsize = (16,6))
        #plt.show()
        print "Optimized Parameters for the {} Classifier for f1_score are".format(key)
        print(f1_optimized)
        print "Optimized Scores are"
        print(f1_optimized_scores)
        
        return optimized_clfs
    

def optimize_classifiers(all_data, optimized_clfs):

    # OPTIMIZING THE KNEIGHBORS, LINEAR SUPPORT VECTOR AND MULTI LAYER PERCEPTRON CLASSIFIERS TO F1-SCORE AND RECALL
    # OVER THE GIVEN PARAMETER SPACE AND OVER THE NUMBER OF FEATURES USING THE TOP 20 FEATURES RANKED 
    # ACCORDING TO CHI2 SCORE
    # RECALL FOR MORTALITY PREDICTION IS IMPORTANT IN THAT WE DON'T WANT FALSE NEGATIVES. 

    # NEAREST NEIGHBOR CLASSIFIER PARAMETERS

    kneighbors_params = {'n_neighbors': [2,4,6,8,10,12], 
                         'weights': ['uniform', 'distance'], 
                         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        }
    #LinearSVC CLASSIFIER PARAMETERES
    LSVC_params = {'C':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1],  
                   'class_weight': [#{1:1, 0:1}, {1:2, 0:1}, {1:2.5, 0:1}, 
                                    {1:3, 0:1}, {1:3.5, 0:1}, {1:4, 0:1}, 
                                    {1:4.5, 0:1}, {1:5, 0:1}, {1:5.5, 0:1}, {1:6, 0:1}], 
                                    #{1:6, 0:1}, {1:6.7, 0:1}, {1:7, 0:1}],
                       'loss':['hinge', 'squared_hinge']
                  }


    MLP_params = {
                  'activation': ['identity', 'logistic', 'tanh', 'relu'], 
                  'solver': ['lbfgs', 'sgd', 'adam']
             
                 }

    Tree_params = {
                   'class_weight': [{1:1, 0:1}, {1:2, 0:1}, {1:2.5, 0:1}, 
                                    {1:3, 0:1}, {1:3.5, 0:1}, {1:4, 0:1}, 
                                    {1:4.5, 0:1}, {1:5, 0:1}, {1:5.5, 0:1}, {1:6, 0:1}],
                    'criterion': ['gini', 'entropy']
                  }



    # DICTIONARY OF CLASSIFIERS AND CORRESPONDING PARAMETERS 


    # CLASSIFIER SCORES
    scores = ['f1', 'recall'] #'accuracy',, 'recall' , 'precision']

    # DICTIONARY OF CLASSIFIERS AND CORRESPONDING PARAMETERS 
    Kneighb = KNeighborsClassifier()
    LSVC = svm.LinearSVC(random_state = 42)
    MLP = MLPClassifier(random_state = 42)
    Tree = DecisionTreeClassifier(random_state = 42) 
    #SVC = svm.SVC(random_state = 42)
    LSVC_stuff = [LSVC, LSVC_params]
    KNeighbors_stuff = [Kneighb, kneighbors_params]
    MLP_stuff = [MLP, MLP_params]
    Tree_stuff = [Tree, Tree_params]

    #SVC_stuff = [SVC, SVC_params]
    classifiers = {'LSVC': LSVC_stuff,
                   'Kneighbors': KNeighbors_stuff, 
                   'MLP': MLP_stuff,
                   'Tree': Tree_stuff
               
                  }
                


    for key in classifiers.keys():
    
        recall_mortality = -100000
        f1_mortality = -100000
        optimized_params = {}
        optimized = {}
    
        optimized_scores = ""
        recall_scores = {}
        f1_scores = {}

        print "Evaluating {}".format(key)
        for num_feats in range(8, all_data.shape[1]):
            print "{} features".format(num_feats)
            # FEATURES AND TARGETS
            X_best = all_data[all_data.columns[1:(num_feats+1)]]
            y = all_data['hospital_expire_flag']

            X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.30, random_state = 42)
        
            if key == 'MLP':
                input_layer = num_feats
                middle_layer = int(math.ceil(num_feats/2))
                classifiers[key][0] = MLPClassifier(random_state = 42, 
                                                    hidden_layer_sizes = (input_layer, middle_layer)
                                                    )
                print "MLP Layers ({}, {})".format(num_feats, math.ceil(num_feats/2))
            # MAY NOT BE NECESSARY THE WAY THE CODE IS CURRENTLY BROKEN OUT
            for score in scores:
                #print"# Optimizing parameters for {} classifier to {} score using {} features".format(key, score, num_feats)

         
                clf = GridSearchCV(classifiers[key][0], classifiers[key][1], cv = 5, 
                               scoring = score)

                clf.fit(X_train, y_train)
    
                y_true, y_pred = y_test, clf.predict(X_test)
                scores_report = metrics.classification_report(y_true, y_pred)
                #print(scores_report)
         
                if score == 'recall':
                    surv, mort = metrics.recall_score(y_true, y_pred, average = None)
                    # a metric that maximizes both variables and minimizes the difference between them
                    recall_metric = mort + surv - (abs(mort-surv))
                    recall_scores[str(num_feats)] = [surv, mort, recall_metric]
                    if (recall_metric > recall_mortality):
                        recall_mortality = recall_metric 
                        optimized = clf.best_params_
                        optimized['features'] = num_feats
                        optimized_scores = scores_report
                        recall_str = key + '_recall'
                        if recall_str not in optimized_clfs:
                            optimized_clfs[recall_str] = {}
                        optimized_clfs[recall_str]['CLF'] = clf.best_estimator_
                        optimized_clfs[recall_str]['PARAMS'] = optimized
                        optimized_clfs[recall_str]['SCORES'] = optimized_scores
                        #print "OPTIMIZED RECALL VALUES REGISTERED"
                    
         
                elif score == 'f1':
            
                    surv_f1, mort_f1 = metrics.f1_score(y_true, y_pred, average = None)
                    #a metric that maximizes both variables and minimizes the difference between them
                    f1_metric = surv_f1 + mort_f1 - (abs(surv_f1-mort_f1))
                    f1_scores[str(num_feats)] = [surv_f1, mort_f1, f1_metric]
                    if (f1_metric > f1_mortality):
                        f1_mortality = f1_metric #math.sqrt(mort**2 + surv**2)
                        #recall_delta = abs(mort-surv)
                        f1_optimized = clf.best_params_
                        f1_optimized['features'] = num_feats
                        f1_optimized_scores = scores_report
                        f1_str = key + '_f1'
                        if f1_str not in optimized_clfs:
                            optimized_clfs[f1_str] = {}
                        optimized_clfs[f1_str]['CLF'] = clf.best_estimator_
                        optimized_clfs[f1_str]['PARAMS'] = f1_optimized
                        optimized_clfs[f1_str]['SCORES'] = f1_optimized_scores
                        #print "OPTIMIZED F1 VALUES REGISTERED"
                    
            
   
        print "ANALYSIS COMPLETE"
        recalls = pd.DataFrame.from_dict(recall_scores)
        recalls = recalls.transpose()
        recalls['features'] = recalls.index
        recalls['features'] = recalls['features'].apply(lambda x: float(x))
        recalls.sort_values(by = 'features', inplace = True)
        recalls.columns = ['Survivors', 'Non-Survivors', 'Metric', 'Features']
        #recalls[['Survivors', 'Non-Survivors', 'Metric']].plot(title = 'Recall Scores', 
        #                                                       use_index = True, figsize = (16,6))
        #plt.show()
        print "Optimized Parameters for recall are"
        print(optimized)
        print "optimized Scores are"
        print(optimized_scores)
        
    
        f1s = pd.DataFrame.from_dict(f1_scores)
        f1s = f1s.transpose()
        f1s['features'] = f1s.index
        f1s['features'] = f1s['features'].apply(lambda x: float(x))
        f1s.sort_values(by = 'features', inplace = True)
        f1s.columns = ['Survivors', 'Non-Survivors', 'Metric', 'Features']
        #f1s[['Survivors', 'Non-Survivors', 'Metric']].plot(title = 'F1 Scores', use_index = True, figsize = (16,6))
        #plt.show()
        print "Optimized Parameters for f1 are"
        print(f1_optimized)
        print "Optimized Scores are"
        print(f1_optimized_scores)
    
    print "ANALYSIS COMPLETE"
    return optimized_clfs


def optimize_splits(all_data, optimized_clfs):
    # TRAINING AND TESTING CLASSIFIERS PREVIOUSLY OPTIMIZED OVER PARAMETER SPACE AND FEATURES 
    # OVER A RANGE OF TRAIN TEST SPLIT SIZE. 

    for key in optimized_clfs.keys():
    
        recall_mortality = 0
        f1_mortality = 0
        optimized_params = {}
        optimized = {}
        optimized_scores = ""
        recall_scores = {}
        f1_scores = {}

        print "Evaluating {}".format(key)
        for testes_size in np.linspace(0.1, 0.5, 9):
            # DEFINING MLP HERE SO THAT WE CAN ALTER THE NUMBER OF INPUT HIDDEN NODES TO MATCH FEATURES
        
            #print "{} test size".format(testes_size)
            # FEATURES AND TARGETS
            features = optimized_clfs[key]['PARAMS']['features']
            clf = optimized_clfs[key]['CLF']
            X_best = all_data[all_data.columns[1:features]]
            y = all_data['hospital_expire_flag']

            X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = testes_size, random_state = 42)
        
        
        
            clf.fit(X_train, y_train)
    
            y_true, y_pred = y_test, clf.predict(X_test)
            scores_report = metrics.classification_report(y_true, y_pred)
                #print(scores_report)
         
            if 'recall' in key:
                surv, mort = metrics.recall_score(y_true, y_pred, average = None)
                    # a metric that maximizes both variables and minimizes the difference between them
                recall_metric = mort + surv - (abs(mort-surv))
                recall_scores[str(testes_size)] = [surv, mort, recall_metric]
                if (recall_metric > recall_mortality):
                    recall_mortality = recall_metric #math.sqrt(mort**2 + surv**2)
                    optimized = optimized_clfs[key]['PARAMS']
                    optimized['test_size'] = testes_size
                    optimized_clfs[key]['PARAMS']['test_size'] = testes_size
                    optimized_clfs[key]['SCORES'] = scores_report
                    optimized_scores = scores_report
                
         
            elif 'f1' in key:
                
                surv_f1, mort_f1 = metrics.f1_score(y_true, y_pred, average = None)
                    #a metric that maximizes both variables and minimizes the difference between them
                f1_metric = surv_f1 + mort_f1 - (abs(surv_f1-mort_f1))
                f1_scores[str(testes_size)] = [surv_f1, mort_f1, f1_metric]
                if (f1_metric > f1_mortality):
                    f1_mortality = f1_metric #math.sqrt(mort**2 + surv**2)
                    #recall_delta = abs(mort-surv)
                    f1_optimized = optimized_clfs[key]['PARAMS']
                    f1_optimized['test_size'] = testes_size
                    optimized_clfs[key]['PARAMS']['test_size'] = testes_size
                    optimized_clfs[key]['SCORES'] = scores_report
                    f1_optimized_scores = scores_report
                
            
        if 'recall' in key:   
            print "ANALYSIS COMPLETE"
            recalls = pd.DataFrame.from_dict(recall_scores)
            recalls = recalls.transpose()
            recalls['test_size'] = recalls.index
            recalls['test_size'] = recalls['test_size'].apply(lambda x: float(x))
            recalls.sort_values(by = 'test_size', inplace = True)
            recalls.columns = ['Survivors', 'Non-Survivors', 'Metric', 'test_size']
            '''
            recalls[['Survivors', 'Non-Survivors', 'Metric']].plot(title = key + ': Recall Scores', 
                                                               use_index = True, figsize = (16,6))
            plt.show()
            '''
            print "Optimized Parameters for {} recall-score are".format(key)
            print(optimized)
            print "optimized Scores are"
            print(optimized_scores)
    
    
        elif 'f1' in key:
            f1s = pd.DataFrame.from_dict(f1_scores)
            f1s = f1s.transpose()
            f1s['test_size'] = f1s.index
            f1s['test_size'] = f1s['test_size'].apply(lambda x: float(x))
            f1s.sort_values(by = 'test_size', inplace = True)
            f1s.columns = ['Survivors', 'Non-Survivors', 'Metric', 'test_Size']
            '''
            f1s[['Survivors', 'Non-Survivors', 'Metric']].plot(title = key + ': F1 Scores', 
                                                        use_index = True, figsize = (16,6))
            plt.show()
            '''
            print "Optimized Parameters for {} f1-score are".format(key)
            print(f1_optimized)
            print "Optimized Scores are"
            print(f1_optimized_scores)
    
    print "OPTIMIZATION COMPLETE"


def write_optimized_clfs(optimized_clfs):
    
    with open('../data/Optimized_Classifiers.txt', 'w') as file:
        file.write(pickle.dumps(optimized_clfs))
    print "Optimized Classifiers Written"
    
def read_optimized_clfs():
    # READ IN PREVIOUSLY CALCULATED OPTIMIZED PARAMETERS
    optimized_clfs = pickle.load(open('../data/Optimized_Classifiers.txt', 'rb'))
    return optimized_clfs
    


def optimized_scores_report(optimized_clfs):
    # CREATE DATAFRAME FROM SCORE REPORTS FOR OPTIMIZED CLASSIFIERS

    first = True
    for key in optimized_clfs.keys():
        print(key)
        #print(optimized_clfs[key]['SCORES'])

        # CONVERT SCORES_REPORT TO DATAFRAME
        scores = optimized_clfs[key]['SCORES'].split()
        #display(scores)
        cols = scores[:4]
        '''
        ind = [x for x in scores if scores.index(x) in [4,9,14, 15, 16]]
        ind[2] = ind[2] + ind[3] + ind[4] 
        ind.pop()
        ind.pop()
        '''
        ind = ['Survivors', 'Non-Survivors', 'Avg/Total']
    
        # CLEAN THIS UP
        dat = [[float(x) for x in scores[5:9]], [float(x) for x in scores[10:14]], 
                                                [float(x) for x in scores[17:22]]]
        #display(cols)
        #display(ind)
        #display(dat)
    
        arrays = [[key, key, key], ind]
        tuples = list(zip(*arrays))
        mindex = pd.MultiIndex.from_tuples(tuples, names=['Classifier', 'Classes'])
        if first:
            scores_frame = pd.DataFrame(dat, columns = cols, index = mindex)
            first = False
        else: 
            scores_frame = scores_frame.append(pd.DataFrame(dat, columns = cols, index = mindex))

    scores_frame.sort_index(level = 0)
    return scores_frame
    
def write_scores_to_file(scores_frame):
    # WRITING OPTIMIZED SCORES TO FILE
    scores_frame.to_csv('Optimized_Classifier_Scores.csv')


def plot_optimized_scores(scores_frame, save_flag):

    monkey = scores_frame.copy()
    classers = list(monkey.index.levels[0])
    metric_dict = {}

    monkey['f1 metric'] = 0.00
    monkey['recall metric'] = 0.00

    for clfr in classers:
        f1survs = monkey['f1-score'][clfr, 'Survivors']
        f1deads = monkey['f1-score'][clfr, 'Non-Survivors']
        f1met = (f1survs+f1deads)-abs(f1survs-f1deads)
        recall_survs = monkey['recall'][clfr, 'Survivors']
        recall_deads = monkey['recall'][clfr, 'Non-Survivors']
        recallmet = (recall_survs+recall_deads)-abs(recall_survs-recall_deads)
        #display(recallmet)
        metric_dict[clfr] = [f1met, recallmet]
        monkey['f1 metric'][clfr, 'avg/total'] = f1met
        monkey['recall metric'][clfr, 'avg/total'] = recallmet

    #display(monkey)
    metric_df = pd.DataFrame.from_dict(metric_dict)
    metric_df.index = ['F1 Metric', 'Recall Metric']
    metric_df = metric_df.transpose()
    display(metric_df)

    ax1 = metric_df.plot.bar(legend = True, figsize = (10,7), rot = 60)
    #ax2 = metric_df['Recall Metric'].plot(secondary_y=True, label = 'Recall Metric', legend = True, rot = 60)

    plt.title('Optimization Metric Scores for Classifiers with Optimized Parameters\n Feature Set Size and Train/Test Splits')

    plt.subplots_adjust(left = 0.1, bottom = 0.25, 
                    right = 0.9, top = 0.9) 
    plt.title('Benchmark Optimization Metric Scores for Classifiers with Default Parameters\n Trained and Tested on 20 Features')
    
    if save_flag:
        save_file_name1 =  "../figures/classifier_benchmarks.png"
   
        print "saving {}".format(save_file_name1)
        plt.savefig(save_file_name1)
    
    plt.show()
    #plt.close()
    
def compare_optimized_to_benchmark(all_data, 

    # CALCULATING CROSS VALIDATION SCORES FOR HIGHEST PERFORMING BENCHMARK CLASSIFIER AND 
    # HIGHEST PERFORMING OPTIMIZED CLASSIFIER

    cross_val_dict = {}

    print "Benchmark Decision-Tree Classifier Performance"
    Tree = DecisionTreeClassifier() 
    X_default = all_data[all_data.columns[1:]]
    y_default = all_data['hospital_expire_flag']

    X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.20, random_state = 42)

    #print "Cross validation score on f1 (Non-Survivors Only)"
    default_scores = cross_val_score(Tree, X_train, y_train, cv = 5, scoring = 'f1')
    cross_val_dict['F1 Non-Surivors: Default DecisionTree'] = default_scores
    #display(default_scores) 
    #print "Cross validation score on f1 (Average for Survivors and Non-Survivors)"
    default_scores = cross_val_score(Tree, X_train, y_train, cv = 5, scoring = 'f1_macro')
    cross_val_dict['F1 Average: Default DecisionTree'] = default_scores
    #display(default_scores) 

    #print "Cross validation score on recall (Non-Survivors Only)"
    default_scores = cross_val_score(Tree, X_train, y_train, cv = 5, scoring = 'recall')
    cross_val_dict['Recall Non-Surivors: Default DecisionTree'] = default_scores
    #display(default_scores)
    #print "Cross validation score on recall (Average for Survivors and Non-Survivors)"
    default_scores = cross_val_score(Tree, X_train, y_train, cv = 5, scoring = 'recall_macro')
    cross_val_dict['Recall Average: Default DecisionTree '] = default_scores
    #display(default_scores)

                                
                                
    #print "Tree fit and test score"

    #display(Tree.score(X_test, y_test))
    Tree.fit(X_train, y_train)
    print "Tree Confusion Matrix and Classification Report"
    y_preds = Tree.predict(X_test)
    display(metrics.confusion_matrix(y_test, y_preds))
    print metrics.classification_report(y_test, y_preds)


    # PERFORMING CROSS VALIDATION ON LSVC_RECALL WITH OPTIMIZED PARAMETERS
    print "Optimized LSVC Classifier Performance"
    X_best = all_data[all_data.columns[1:11]]
    y = all_data['hospital_expire_flag']
    X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.50, random_state = 42)

    clf = optimized_clfs['LSVC_recall']['CLF']#.best_estimator_

    #print "Cross validation score on f1 (Non-Survivors Only)"
    optimized_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')
    cross_val_dict['F1 Non-Surivors: Optimized LSVC'] = optimized_scores
    #display(optimized_scores)
    #print "Cross validation score on f1 (Average for Survivors and Non-Survivors)"
    optimized_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1_macro')
    cross_val_dict['F1 Average: Optimized LSVC'] = optimized_scores
    #display(optimized_scores)

    #print "Cross validation score on recall (Non-Survivors Only)"
    optimized_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'recall')
    cross_val_dict['Recall Non-Survivors: Optimized LSVC'] = optimized_scores
    #display(optimized_scores)
    #print "Cross validation score on recall (Average for Survivors and Non-Survivors)"
    optimized_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'recall_macro')
    cross_val_dict['Recall Average: Optimized LSVC'] = optimized_scores
    #display(optimized_scores)


    #print "LSVC fit test"
    #display(scores)  

    #display(clf.score(X_test, y_test))

    clf.fit(X_train, y_train)
    print "LSVC Confusion Matrix and Classification Report"
    y_preds = clf.predict(X_test)
    display(metrics.confusion_matrix(y_test, y_preds))
    print metrics.classification_report(y_test, y_preds)

    cross_val_df = pd.DataFrame.from_dict(cross_val_dict)
    cross_cols = list(cross_val_df.columns)
    cross_cols.sort()
    cross_val_F1_df = cross_val_df[cross_cols[:4]]
    cross_val_Recall_df = cross_val_df[cross_cols[4:]]
    display(cross_val_F1_df.round(2))
    display(cross_val_Recall_df.round(2))

def cross_val_scores(data, clf):

    # CALCULATING CROSS VALIDATION SCORES FOR HIGHEST PERFORMING BENCHMARK CLASSIFIER AND 
    # HIGHEST PERFORMING OPTIMIZED CLASSIFIER

    cross_val_dict = {}

    
    X_default = data[data.columns[1:]]
    y_default = data['hospital_expire_flag']

    X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size = 0.20, 
                                                            random_state = 42)

    # calculate cross validation scores for different metrics, f1, f1_macro, 
    # recall and recall_macro
    
    clf_name = type(clf).__name__
    default_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')
    key = 'F1 Non-Surivors: ' +  clf_name
    cross_val_dict[key] = default_scores
    
    default_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1_macro')
    key = 'F1 Average: ' + clf_name
    cross_val_dict[key] = default_scores
    #display(default_scores) 

    #print "Cross validation score on recall (Non-Survivors Only)"
    default_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'recall')
    key = 'Recall Non-Surivors: ' + clf_name
    cross_val_dict[key] = default_scores
    
    default_scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'recall_macro')
    key = 'Recall Average: ' + clf_name
    cross_val_dict[key] = default_scores
    #display(default_scores)

                        
                        
    #print "Tree fit and test score"

    #display(Tree.score(X_test, y_test))
    clf.fit(X_train, y_train)
    print clf_name + " Confusion Matrix and Classification Report"
    y_preds = clf.predict(X_test)
    display(metrics.confusion_matrix(y_test, y_preds))
    print metrics.classification_report(y_test, y_preds)
    return cross_val_dict
    








print "****************************************************"
print "**************** Importing Features and ************"
print "******************** Outcomes **********************"
print "****************************************************"
all_data = import_features()
print "shape of all_data = {}".format(all_data.shape)
#for col in all_data.columns:
#    print col
    

#print(all_data[all_data.columns[:4]].head(3))
print "****************************************************"
print "**************** Splitting Features ****************"
print "****************************************************"
diagnoses_features, meas_features = block_features(all_data)
print "****************************************************"
print "**************** Diagnoses Features ****************"
print "****************************************************"
#for feat in diagnoses_features.columns:
#    print(feat)
    
print "****************************************************"
print "**************** Measures Features *****************"
print "****************************************************"

#for feat in meas_features.columns:
#    print(feat)
    
print "****************************************************"
print "**************** diagnoses features stuff **********"
print "****************************************************"

#monkey =diagnoses_features.groupby('hospital_expire_flag').sum()

#print(monkey.append(diagnoses_features.groupby('hospital_expire_flag').sum()/diagnoses_features.groupby('hospital_expire_flag').count()).transpose())

plot_survival_rates(meas_features, True)
plot_true_rates(meas_features, True)

print "****************************************************"
print "*** Creating Classifier Performance Baseline ****"
print "****************************************************"
def_scores = create_benchmarks(all_data)
print(def_scores.sort_index())
plot_benchmarks(def_scores, True)
optimized_clfs = {}

'''
# optimization parameters were pre-calculated so will be read in 
# from file 

print "****************************************************"
print "*** Optimizing Classifiers Except SVC           ****"
print "****************************************************"
optimized_clfs = optimize_classifiers(all_data, optimized_clfs)
for key in optimized_clfs.keys():
    print key
    
print "****************************************************"
print "***           Optimize SVC                      ****"
print "****************************************************"
optimized_clfs = optimize_SVC(all_data, optimized_clfs)

print "****************************************************"
print "***           Optimize on Train Test Splits     ****"
print "****************************************************"
optimized_clfs = optimize_splits(all_data, optimized_clfs)

'''

optimized_clfs = read_optimized_clfs()
# CREATE CLASSIFIER INSTANCES FOR CALCULATING CROSS VALIDATION SCORES
# USING HIGHEST PERFORMING DEFAULT, TREE CLASSIFIER, AND HIGHEST PERFORMING
# OPTIMIZED, THE LSVC CLASSIFIER .
Tree = DecisionTreeClassifier() 
clf = optimized_clfs['LSVC_recall']['CLF']#.best_estimator_

#LSVC optimized for first 11 features
LSVC_data = all_data[all_data.columns[:11]]
   
    