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
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from heapq import nlargest
from sklearn.model_selection import GridSearchCV


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
        print(name)
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
    
    print "scores"
    print(all_scores)
   
    
    

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
    '''
    meas_list = ['hospital_expire_flag', 'Creat_abnflag','icu_stay_Q3','first_careunit_CSRU',
                              'admission_type_ELECTIVE', 'icu_stay_Q0','age_Q3',
                              'first_careunit_MICU'
                ]
    '''
    diagnoses_features = all_data[diagnoses_list]
    #meas_features = all_data[diagnoses_list]    
    
    return diagnoses_features #, meas_features
    

print "****************************************************"
print "**************** Importing Features and ***********************"
print "******************** Outcomes *******************"
print "****************************************************"
all_data = import_features()
print "shape of all_data = {}".format(all_data.shape)
for col in all_data.columns:
    print col
    

print(all_data[all_data.columns[:4]].head(3))
print "****************************************************"
print "**************** Splitting Features ***********************"
print "****************************************************"
diagnoses_features = block_features(all_data)
monkey =diagnoses_features.groupby('hospital_expire_flag').sum()

print(monkey.append(diagnoses_features.groupby('hospital_expire_flag').sum()/diagnoses_features.groupby('hospital_expire_flag').count()).transpose())


           