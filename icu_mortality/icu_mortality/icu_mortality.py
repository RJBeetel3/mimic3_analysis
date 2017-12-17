import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
