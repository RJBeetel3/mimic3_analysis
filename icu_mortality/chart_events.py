#import sys
import os
import pandas as pd
import datetime as datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt


#from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#from sklearn.feature_selection import f_classif
#from heapq import nlargest




def import_chartevents_data():
    print "importing chart data"
    data = pd.DataFrame.from_csv('../data/CHART_EVENTS_FIRST24.csv')
    #print(data.head())
    print "converting date-time data"
    data.loc[:,'charttime']  = pd.to_datetime(data.loc[:,'charttime']) 
    data['subject_id'] = data.index
    print "reorganizing data"
    data.set_index(np.arange(data.shape[0]), inplace = True)
    cols = list(data.columns)
    cols.insert(0, cols.pop(cols.index('icustay_id')))
    cols.insert(1, cols.pop(cols.index('subject_id')))
    data = data[cols]
    icu_stays = data.drop_duplicates('icustay_id', keep = 'first').shape[0]
    patients = data.drop_duplicates('subject_id', keep = 'first').shape[0]
    print "The number of chart events = {}".format(data.shape)
    print "The number of unique ICU stays = {}".format(icu_stays)
    print "The number of unique patients  = {}".format(patients)
    print "chart data import complete"
    return data    


def explore_data(data):
    # display the different measurements captured in the database query
    labels = list(data.label.unique())
    #print "the measurements included in chart events are as follows:"
    #for measurement in labels:
    #    print(measurement) 
    labels.sort()
    
    '''
    # CODE FOR PRINTING THE NUMBER OF SAMPLES FOR EACH MEASUREMENT
    for item in labels:
        print "the number of samples for {} is {}".format(item, 
                data['icustay_id'][data.label == item].dropna().unique().shape[0])
    '''
    
    
    
    # REMOVE ALL VARIABLES WITH FEWER THAN 2000 SAMPLES
    old_cols = [x for x in labels if (data['icustay_id'][data.label == x].dropna().unique().shape[0] >= 2000)]
    print "There are {} measurements having > 2k samples".format(len(old_cols))
    #CREATE LISTS FOR CONSTANT CATEGORICAL AND CONTINOUS DATA
    #CONSTANT VARIABLES INCLUDE ADMISSION WEIGHT, HEIGHT
    old_cols_const = [old_cols[x] for x in [0, 14]]
    print "contstant variables: \n ********************"
    print old_cols_const
    #CATEGORICAL VARIABLES INCLUDE GLASGOW COMA SCALE (GSC)
    # AND CAPILLARY REFILL
    old_cols_cat = [x for x in old_cols if 'GCS' in x]
    old_cols_cat.append(old_cols[6])
    print "categorical variables: \n ********************"
    print old_cols_cat
    # create list for continuous variables
    old_cols_continuous = [x for x in old_cols if ((x not in old_cols_const) & (x not in old_cols_cat))]
    print "continuous variables: \n ********************"
    print old_cols_continuous
    return old_cols_continuous, old_cols_const, old_cols_cat


def calculate_stats(data, old_cols_continuous, old_cols_const, old_cols_cat):
    
    # *** CODED MORE CONCISELY IN lab_events.py *** 


    # create dictionaries for constant and categorical data
    const_dict = {}
    cat_dict = {}

    # create dictionaries in which measurements will be organized and stored

    mean_dict = {}
    med_dict = {}
    std_dict = {}
    skew_dict = {}
    min_dict = {}
    max_dict = {}
    first_dict = {}
    slope_dict = {}
    delta_dict = {}

    calc_dict_cols = ['pH2', 'BP_Mean', 'BP_Dia', 'BP_Sys', 'pH3', 'Creat2', 'GlucC', 'HR', 'Hemat','Hg', 'O2_Fraction', 
                   'RR_Spont','RR_Total', 'RR', 'TempC', 'TempC_Calc']
    const_dict_cols = ['Weight', 'Height', 'O2_Fraction', 'RR', 'TempC', 'TempC_Calc']
    cat_dict_cols = ['GCS_Eye', 'GCS_Motor','GCS_Verbal', 'GCS_total', 'Cap_refill' ]


    mean_dict_new_cols = []
    med_dict_new_cols = []
    std_dict_new_cols = []
    skew_dict_new_cols = []
    min_dict_new_cols = []
    max_dict_new_cols = []
    first_dict_new_cols = []
    slope_dict_new_cols = []
    delta_dict_new_cols = []
    for x in calc_dict_cols:
        mean_dict_new_cols.append(x + '_mean')
        med_dict_new_cols.append(x + '_med')
        std_dict_new_cols.append(x + '_std')
        skew_dict_new_cols.append(x + '_skew')
        min_dict_new_cols.append(x + '_min')
        max_dict_new_cols.append(x + '_max')
        first_dict_new_cols.append(x + '_first')
        slope_dict_new_cols.append(x + '_slope')
        delta_dict_new_cols.append(x + '_delta')


    # height and weight are left out from the calculated measures because there was only one
    # measurement so they are constant.

    # IF I SWITCH THE ORDER OF THESE, I CAN MAKE THE OLD COLS KEYS FOR EACH DICT, THEN 
    # I CAN LOOP THROUGH THE OLD COLS AND DO CALCULATIONS FOR EACH DICT RATHER THAN ITERATING THROUGH 
    # THE COLUMNS FOR EACH 
    mean_dict_names = dict(zip(mean_dict_new_cols, old_cols_continuous))
    med_dict_names = dict(zip(med_dict_new_cols, old_cols_continuous))
    std_dict_names = dict(zip(std_dict_new_cols, old_cols_continuous))
    skew_dict_names = dict(zip(skew_dict_new_cols, old_cols_continuous))
    min_dict_names = dict(zip(min_dict_new_cols, old_cols_continuous))
    max_dict_names = dict(zip(max_dict_new_cols, old_cols_continuous))
    first_dict_names = dict(zip(first_dict_new_cols, old_cols_continuous))
    slope_dict_names = dict(zip(slope_dict_new_cols, old_cols_continuous))
    delta_dict_names = dict(zip(delta_dict_new_cols, old_cols_continuous))
    const_dict_names = dict(zip(const_dict_cols, old_cols_const))
    cat_dict_names = dict(zip(cat_dict_cols, old_cols_cat))
    #display(mean_dict_names)
    #display(const_dict_names)
    #display(cat_dict_names)

    print "dict_name creation complete"

    # COULD POSSIBLY WRAP ALL THIS UP AND ITERATE BUT THAT MIGHT BE TOO CONFUSING....

    # ITERATING THROUGH THE VARIABLES, CALCULATING MEANS, MEDIANS, STD, SKEWNESS, MIN AND MAX'S FOR EACH ITERATION
    # COME BACK AND REFINE THIS SO THAT THE DATA COLUMN NAMES ARE THE DICTIONARY KEYS, THEN WE CAN JUST ITERATE 
    # THROUGH THOSE AND DO CALCULATIONS FOR EACH DICT IN A SINGLE LOOP
    # ** CAN BE REPRESENTED MORE CONCISELY, SEE LABEVENTS_FIRST24.ipynb ** 
    print "calculating mean values"
    for col in mean_dict_names.keys():
        mean_dict[col] = pd.DataFrame(data[data.label == mean_dict_names[col]].groupby('icustay_id')['valuenum'].mean())
        mean_dict[col].columns = [mean_dict_names[col]]
        mean_dict[col]['hospital_expired_flag'] = data[data.label == mean_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        mean_dict[col]['gender'] = data[data.label == mean_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, mean_dict[col].shape)
    print "calculating med values"
    for col in med_dict_names.keys():
        med_dict[col] = pd.DataFrame(data[data.label == med_dict_names[col]].groupby('icustay_id')['valuenum'].median())
        med_dict[col].columns = [med_dict_names[col]]
        med_dict[col]['hospital_expired_flag'] = data[data.label == med_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        med_dict[col]['gender'] = data[data.label == med_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, med_dict[col].shape)
    print "calculating std values"
    for col in std_dict_names.keys(): 
        std_dict[col] = pd.DataFrame(data[data.label == std_dict_names[col]].groupby('icustay_id')['valuenum'].std())
        std_dict[col].columns = [std_dict_names[col]]
        std_dict[col]['hospital_expired_flag'] = data[data.label == std_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        std_dict[col]['gender'] = data[data.label == std_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, std_dict[col].shape)
    print "calculating skewness values"
    for col in skew_dict_names.keys(): 
        skew_dict[col] = pd.DataFrame(data[data.label == skew_dict_names[col]].groupby('icustay_id')['valuenum'].skew())
        skew_dict[col].columns = [skew_dict_names[col]]
        skew_dict[col]['hospital_expired_flag'] = data[data.label == skew_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        skew_dict[col]['gender'] = data[data.label == skew_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, skew_dict[col].shape)
    print "calculating min values"
    for col in min_dict_names.keys():   
        min_dict[col] = pd.DataFrame(data[data.label == min_dict_names[col]].groupby('icustay_id')['valuenum'].min())
        min_dict[col].columns = [min_dict_names[col]]
        min_dict[col]['hospital_expired_flag'] = data[data.label == min_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        min_dict[col]['gender'] = data[data.label == min_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, min_dict[col].shape)
    print "calculating max values"
    for col in max_dict_names.keys():       
        max_dict[col] = pd.DataFrame(data[data.label == max_dict_names[col]].groupby('icustay_id')['valuenum'].max())
        max_dict[col].columns = [max_dict_names[col]]
        max_dict[col]['hospital_expired_flag'] = data[data.label == max_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        max_dict[col]['gender'] = data[data.label == max_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, max_dict[col].shape)

    print "extracting first measurements"
    for col in first_dict_names.keys():    
        first_dict[col] = pd.DataFrame(data[data.label == first_dict_names[col]].groupby('icustay_id')['valuenum'].first())
        first_dict[col].columns = [first_dict_names[col]]
        first_dict[col]['hospital_expired_flag'] = data[data.label == first_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        first_dict[col]['gender'] = data[data.label == first_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, first_dict[col].shape)

    print "calculating delta"
    for col in delta_dict_names.keys():
        delta_dict[col] = pd.DataFrame(data[data.label == delta_dict_names[col]].groupby('icustay_id')['valuenum'].last() - 
                                       data[data.label == delta_dict_names[col]].groupby('icustay_id')['valuenum'].first())
        delta_dict[col].columns = [delta_dict_names[col]]
        delta_dict[col]['hospital_expired_flag'] = data[data.label == delta_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        delta_dict[col]['gender'] = data[data.label == delta_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, delta_dict[col].shape)

    print "calculating slope"
    for col in slope_dict_names.keys():
        val_last = data[data.label == slope_dict_names[col]].groupby('icustay_id')['valuenum'].last()  
        val_first = data[data.label == slope_dict_names[col]].groupby('icustay_id')['valuenum'].first()
        time_last = data[data.label == slope_dict_names[col]].groupby('icustay_id')['charttime'].last()  
        time_first = data[data.label == slope_dict_names[col]].groupby('icustay_id')['charttime'].first()
        slope_dict[col] = pd.DataFrame((val_last - val_first)/((time_last - time_first)/np.timedelta64(1,'h')))  
        slope_dict[col].columns = [slope_dict_names[col]]
        slope_dict[col]['hospital_expired_flag'] = data[data.label == slope_dict_names[col]].groupby('icustay_id').hospital_expire_flag.first()
        slope_dict[col]['gender'] = data[data.label == slope_dict_names[col]].groupby('icustay_id').gender.first()
        #print "{} number of samples = {}".format(col, slope_dict[col].shape)


    print "Summary Calculations Complete"

    for col in const_dict_names.keys():

        dummy = data[data.label == const_dict_names[col]].groupby('icustay_id')
        const_dict[col] = pd.DataFrame(dummy.valuenum.first())
        const_dict[col].columns = [const_dict_names[col]]
        const_dict[col]['hospital_expired_flag'] = dummy.hospital_expire_flag.first()
        const_dict[col]['gender'] = dummy.gender.first()


    # GCS MEASURES DO HAVE CORRESPONDING VALUENUMS AS CATEGORIES. WILL NOT INCLUDE PRESENTLY
    for col in cat_dict_names.keys():
        dummy = data[data.label == cat_dict_names[col]].groupby('icustay_id')
        cat_dict[col] = pd.DataFrame(dummy.value.first()) 
        cat_dict[col].columns = [cat_dict_names[col]]
        cat_dict[col]['hospital_expired_flag'] = dummy.hospital_expire_flag.first()
        cat_dict[col]['gender'] = dummy.gender.first()


    print "Categorical Dataframes Complete"

    '''    
    calc_dicts = [mean_dict, med_dict, std_dict, skew_dict, min_dict, max_dict, first_dict, 
                 slope_dict, delta_dict]
    '''
    calc_dicts = {'means': mean_dict, 
               'medians': med_dict, 
               'STD' : std_dict, 
               'skewness': skew_dict, 
               'minimum' : min_dict, 
               'maximum' : max_dict, 
               'first' : first_dict, 
               'slope' : slope_dict, 
               'delta' : delta_dict
               }
    
    names_dict = {}
    suffix = '_outliers'

    # SETTING OUTLIER DATA POINTS TO NAN FOR REMOVAL LATER USING DROPNA()
    for frame in calc_dicts.keys():
        for col in calc_dicts[frame].keys():
        # plot
        # print col
            dummy = calc_dicts[frame][col]
            col2 = dummy.columns[0]
            #print "{}   {}     {}".format(col, col2, dummy.dropna().shape)
            Q1 = np.percentile(dummy[col2].dropna(), 25)
            # TODO: Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile(dummy[col2].dropna(), 75)
            # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            step = 1.5*(Q3 - Q1)
            names_dict[col+suffix] = dummy[~((dummy[col2] >= Q1 - step) & (dummy[col2] <= Q3 + step))].index
            dummy.set_value(names_dict[col+suffix], col2, np.NaN)
            #print "{}   {}     {}".format(col, col2, dummy.dropna().shape)



    print "Outlier Removal Complete"


    print "Complete"
    return calc_dicts, const_dict, cat_dict




def remove_non_variable_features(calc_dicts):

    # REMOVE FRAMES/VARIABLES FOR WHICH THERE IS ONLY ONE VALUE I.E. SINGULAR
    for frame in calc_dicts.keys():
        for col in calc_dicts[frame].keys():
            col2 = calc_dicts[frame][col].keys()[0]
            unique_vals = len(calc_dicts[frame][col][col2].dropna().unique())
            if unique_vals < 2:
                print "removing due to only one value  = {}".format(col)
                calc_dicts[frame].pop(col) 



def plot_feature_density(dummy, col, save_flag):

    x25 = dummy[col].dropna().quantile(0.25)
    x50 = dummy[col].dropna().quantile(0.50)
    x75 = dummy[col].dropna().quantile(0.75)
    
    plt.subplots(figsize=(10,6))
    print "plotting feature {}".format(col)
    dummy[dummy.hospital_expired_flag==1][col].dropna().plot.kde(
            alpha=1.0,label='Non-survival')
    dummy[dummy.hospital_expired_flag==0][col].dropna().plot.kde(
            alpha=1.0,label='Survival')

    plt.subplots_adjust(left = 0.1, bottom = 0.1, 
                    right = 0.9, top = 0.9)



    plt.axvline(x = x25, color='k', linestyle='-')
    plt.text(x25+0.05,0.05,'Q1',rotation=0)
    plt.axvline(x = x50, color = 'k', linestyle = '-')
    plt.text(x50+0.05,0.05,'Q2',rotation=0)
    plt.axvline(x = x75, color = 'k', linestyle = '-')
    plt.text(x75+0.05,0.05,'Q3 ',rotation=0)

    plt.title('Mean Temperature Measurement Distributions for Survivors and Non-Survivors ')
    plt.xlabel(col)
    plt.legend(loc="upper left", bbox_to_anchor=(0.75,0.75),fontsize=12)
    plt.show()
    if save_flag:
        save_file_name = "../figures/" + col + "_" + gend + "_" + "plot.png"
        print "saving {}".format(save_file_name)
        plt.savefig(save_file_name)
        
    #plt.close()




def plot_categorical_features(cat_dict, save_flag):
    for col in cat_dict.keys():
        col2 = cat_dict[col].keys()[0]
        #print col
        #print col2
        vals = list(cat_dict[col][col2].unique())
        #display(vals)
        total = cat_dict[col].groupby(col2)[col2].count()
   
        dead = cat_dict[col][cat_dict[col].hospital_expired_flag == 1].groupby(col2)[col2].count()
        dead.name = 'Survivors'
        dead_percent = 100.00*(dead / total)
        live = cat_dict[col][cat_dict[col].hospital_expired_flag == 0].groupby(col2)[col2].count()
        live.name = 'Non_Survivors'
        live_percent = 100.00*(live / total)
        monkey = pd.concat([live_percent, dead_percent], axis = 1)

        #display(monkey)
        plt.subplots(figsize=(10,6))
        print "plotting feature {}".format(col)
        
        monkey.plot.bar(stacked = True, figsize = (10,6), edgecolor = 'black', linewidth = 3, 
                                    alpha = 0.5, title = "Survival Rate for " + col)
        
        plt.subplots_adjust(left = 0.1, bottom = 0.3, 
                    right = 0.9, top = 0.9)    
        plt.xticks(rotation = 15, ha = 'right')  
                          
        if save_flag:
            save_file_name = "../figures/" + col + "_" + col2 + "_" + "plot.png"
            print "saving {}".format(save_file_name)
            plt.savefig(save_file_name)
                                    
        
def plot_continuous_features(frame, save_flag):
    # PLOT MEAN DISTRIBUTION
    for col in frame.keys():
        dummy = frame[col]
        col2 = dummy.columns[0]
       
        plt.subplots(figsize=(10,6))
        print "plotting feature {}".format(col)
        dummy[col2][dummy.hospital_expired_flag==1].dropna().plot.kde(
            alpha=1.0,label='Non-survival')
        dummy[col2][dummy.hospital_expired_flag==0].dropna().plot.kde(
            alpha=1.0,label='Survival')

            # add title, labels etc.
        plt.subplots_adjust(left = 0.1, bottom = 0.1, 
                    right = 0.9, top = 0.9) 
        plt.title('{} measurement on ICU admission '.format(col) +
                    'vs ICU mortality \n')
        plt.xlabel(col)
        plt.legend(loc="upper left", bbox_to_anchor=(0.75,0.75),fontsize=12)
        
        if save_flag:
            save_file_name = "../figures/" + col + "_" + gend + "_"+ "plot.png"
            print "saving {}".format(save_file_name)
            plt.savefig(save_file_name)
        calc_dicts
        
    
    print "complete"
    



def plot_constant_features(const_dict, save_flag):

    # PLOTTING CONSTANT VALUES LIKE HEIGHT AND WEIGHT
    for col in const_dict.keys():

        col2 = const_dict[col].keys()[0]
        vals = list(const_dict[col][col2].unique())

        gender = ['M', 'F'] 

        for gend in gender:
    
            print gend
            dead = const_dict[col][(const_dict[col].hospital_expired_flag == 1)&
                                  (const_dict[col].gender == gend)]
                              #&(const_dict[col][col2] >20)]
            dead.name = 'Non_Survivors'
            live = const_dict[col][(const_dict[col].hospital_expired_flag == 0)&
                                  (const_dict[col].gender == gend)]
                              #&(const_dict[col][col2] >20)]
            live.name = 'Survivors'


            #display(dummy.head())
        
            maxx = 0.99
            minn = 0.01

            live_max = live[col2].quantile(maxx)
            live_min = live[col2].quantile(minn)
            dead_max = dead[col2].quantile(maxx)
            dead_min = dead[col2].quantile(minn)
            maxlim = max(live_max, dead_max)
            minlim = min(live_min, dead_min)


            plt.subplots(figsize=(10,6))
    
       
            live[(live[col2] < live_max) & (live[col2] > live_min)][col2].plot.hist(bins = 100, 
                                                                                alpha=0.3,label='Survivors')

            dead[(dead[col2] < dead_max) & (dead[col2] > dead_min)][col2].plot.hist(bins = 100, 
                                                                                alpha=1.0,label='Non-Survivors')
            # add title, labels etc.
            plt.subplots_adjust(left = 0.1, bottom = 0.1, 
                    right = 0.9, top = 0.9) 
                
            plt.title('{} measurement on ICU admission'.format(col) + 
                       'vs ICU mortality by gender = {}\n'.format(gend))
            plt.xlabel(col)
            plt.legend(loc="upper left", bbox_to_anchor=(0.75,0.75),fontsize=12)


            print "{}    {}".format(maxlim, minlim)
            plt.xlim(minlim, maxlim)
        
            if save_flag:
                save_file_name = "../figures/" + col + "plot.png"
                print "saving {}".format(save_file_name)
                plt.savefig(save_file_name)


def merge_continuous_data(data, calc_dicts, const_dict):

    # MERGE DATAFRAMES HERE 
    # QUESTION UTILITY OF HAVING INDIVIDUAL FRAMES
    # ** CAN BE CODED MORE EFFICIENTLY. SEE LABEVENTS_FIRST24.ipynb ** 
    data2 = data.drop_duplicates('icustay_id', keep = 'first')
    data3 = data2.drop(['label', 'value', 'valuenum'], axis = 1)
    data3.set_index(['icustay_id'], inplace = True)

    for frame in calc_dicts.keys():
        print "Merging {} Values".format(frame)
        for key in calc_dicts[frame].keys():
            col = calc_dicts[frame][key].keys()[0]
            #print "merging {}   {}".format(frame, col)
            #print(calc_dicts[frame][key][col].head())
            data3 = data3.merge(pd.DataFrame(calc_dicts[frame][key][col]), left_index = True, 
                right_index = True, how = 'left', sort = True)
                
            newcols = list(data3.columns)
            newcols.pop()
            newcols.append(key)
            data3.columns = newcols
     
    for col in const_dict.keys():
        col2 = const_dict[col].keys()[0]
        print "merging {}   {}".format(col, col2)
        #print(const_dict[col][col2].head())
        data3 = data3.merge(pd.DataFrame(const_dict[col][col2]), left_index = True, right_index = True, 
                           how = 'left', sort = True)
        newcols = list(data3.columns)
        newcols.pop()
        newcols.append(col)
        data3.columns = newcols
    
            
    data3['icustay_id'] = data3.index
    cols = list(data3.columns)
    cols.sort()
    cols.insert(0, cols.pop(cols.index('icustay_id')))
    cols.insert(1, cols.pop(cols.index('subject_id')))
    cols.insert(2, cols.pop(cols.index('hospital_expire_flag')))


    data3 = data3[cols]
    data3.set_index(np.arange(data3.shape[0]), inplace = True)
    return data3
    

def categorical_to_dummy(data3, cat_dict):
    dummies = data3[data3.columns[:3]]
    dummies.set_index(['icustay_id'], inplace = True)
    
    
    for col in cat_dict.keys():
        col2 = cat_dict[col].keys()[0]
        chimp = pd.get_dummies(cat_dict[col][col2], prefix = cat_dict[col][col2].name)
        dummies = dummies.merge(chimp, left_index = True, right_index = True, 
                           how = 'left', sort = True)


    return dummies


def categorical_affinity_blocks(dummies):
    
    # MEASURES HAVE LOW AFFINITY I.E. WHEN WE DROP NAN VALUES THERE ARE VERY FEW SAMPLES LEFT 
    # SO BREAKING THESE UP INTO HIGH AFFINITY DATAFRAMES FOR PROCESSING. 
    # ** MAY CONSIDER PROCESSING GCS_TOTAL AS A CONTINUOUS BUT, FOR NOW CREATING DUMMIES

    print "Shape of Capillary Block"
    # NUMBER OF NON NAN SAMPLES IN CAPILLARY REFILL
    print(dummies[[x for x in dummies.columns if 'Capillary' in x]].dropna().shape)

    # NUMBER OF NON NAN SAMPLES IN GCS_TOTAL ONLY
    print "Shape of GCS_Total Block"
    print(dummies[[x for x in dummies.columns if 'Total' in x]].dropna().shape)
    # NUMBER OF NON NAN SAMPLES IN GCS MEASURES WITHOUT TOTAL 
    print "Shape of GCS Block"
    print(dummies[[x for x in dummies.columns if (('GCS' in x) & ('Total' not in x))]].dropna().shape)
    # NUMBER OF NON NAN SAMPLES IN GCS TOTAL AND MEASEURES
    print "Shape of All GCS  Block"
    print(dummies[[x for x in dummies.columns if 'GCS' in x]].dropna().shape)
    # NUMBER OF NON NAN SAMPLES IN GCS MEASURES AND CAP REFILL
    print "Shape of Capillary and GCS Block"
    print(dummies[[x for x in dummies.columns if 'Total' not in x]].dropna().shape)

    #CREATE 3 BLOCKS BASED ON AFFINITY I.E. SIZE AFTER NAN VALUES DROPPED
    cap_cols = [x for x in dummies.columns if 'Capillary' in x]
    cap_cols.insert(0, 'hospital_expire_flag')
    Cap_dummies = dummies[cap_cols].dropna()
    GCS_Tot_cols = [x for x in dummies.columns if 'Total' in x]
    GCS_Tot_cols.insert(0, 'hospital_expire_flag')
    GCS_Total_dummies = dummies[GCS_Tot_cols].dropna()
    GCS_cols = [x for x in dummies.columns if (('GCS' in x) & ('Total' not in x))]
    GCS_cols.insert(0, 'hospital_expire_flag')
    GCS_dummies = dummies[GCS_cols].dropna()
    
    cat_dummy_dict = {'Cap_dummies': Cap_dummies, 
                      'GCS_Total_dummies': GCS_Total_dummies, 
                      'GCS_dummies': GCS_dummies
                      }
                      
                      
                          
    #return Cap_dummies, GCS_Total_dummies, GCS_dummies
    return cat_dummy_dict
    
    
    
    
def continuous_affinity_blocks(data3):
    # BREAKING UP VARIABLES SO THAT WE CAN DROP NAN VALUES AND STILL HAVE SUFFICIENT SAMPLES 
    # TO TRANSFORM AND DO FEATURE SELECTION / SCORING
    # WILL NEED TO MERGE LATER IN A WAY THAT PROVIDES ADEQUATE SAMPLES

    cols1 = [x for x in data3.columns if ('BP' in x)]
    cols2 = [x for x in data3.columns if (('Creat2' in x) | ('Gluc' in x) | ('Hg' in x) | ('Hemat' in x) | ('TempC' in x))]
    cols3 = [x for x in data3.columns if ((('RR' in x) & ('Spont' not in x) & ('Total' not in x)) | ('HR' in x))]
    cols4 = [x for x in data3.columns if ('pH' in x)]
    header = ['hospital_expire_flag', 'subject_id', 'icustay_id']
    for thing in header:
        cols1.insert(0, thing)
        cols2.insert(0, thing) 
        cols3.insert(0, thing)
        cols4.insert(0, thing)

    #print(cols1)
    data3.replace([np.inf, -np.inf], np.nan, inplace = True)
    BP_data = data3[cols1].dropna()
    CreatGlucHgHmT_data = data3[cols2].dropna()
    HR_RR_data = data3[cols3].dropna()
    pH_data = data3[cols4].dropna()
                                 
    
    cont_blocks = { 'BP_data': BP_data, 
                    'CreatGlucHgHmT_data' : CreatGlucHgHmT_data, 
                    'HR_RR_data' : HR_RR_data, 
                    'pH_data' : pH_data
                   }
    return cont_blocks



def drop_sparse_data(data3):
    # THE FOLLOWING DROPS COLUMNS WITH SPARSE DATA. THIS WAS DETERMINED IN PREVIOUS ITERATIONS
    # USING AFFINITY MAPS
    # THE COLUMNS BEING DROPPED WERE IDENTIFIED AS SPARSE IN HEATMAPS BELOW
    drop_cols = [x for x in data3.columns if (('O2_Fraction' in x) | (('TempC' in x) & ('TempC_Calc' not in x))) ]
    more_cols = ['Creat2_skew', 'Hg_skew', 'RR_Spont_skew', 'RR_Total_skew']
    for col in more_cols:
        drop_cols.append(col)
    
    '''
    for col in drop_cols:
        print(col)
    print "****************************************************"
    print "*************** DATA 3 Columns !!!******************"
    print "****************************************************"
    print(data3.columns)
    print "****************************************************"
    print "*************** DROP  Columns !!!******************"
    print "****************************************************"
    print(drop_cols) 
    '''
    data3.drop(drop_cols, inplace = True, axis = 1)
    data3.set_index(np.arange(data3.shape[0]), inplace = True)    
    print "Data3 Shape = {}".format(data3.shape)
    return data3
'''   
calc_dict_cols = ['pH2', 'BP_Mean', 'BP_Dia', 'BP_Sys', 'pH3', 'Creat2', 'GlucC', 'HR', 'Hemat','Hg', 'O2_Fraction', 
                   'RR_Spont','RR_Total', 'RR', 'TempC', 'TempC_Calc']
const_dict_cols = ['Weight', 'Height', 'O2_Fraction', 'RR', 'TempC', 'TempC_Calc']
cat_dict_cols = ['GCS_Eye', 'GCS_Motor','GCS_Verbal', 'GCS_total', 'Cap_refill' ]
'''





# CALCULATING THE QUARTILES ON THE DISTRIBUTIONS AND BINNING DATA INTO 4 BUCKETS
# TO CONVERT CONTINUOUS VARIABLES TO CATEGORICAL

def quant_cats(feature, Q1, Q2, Q3):
    if feature <=Q1:
        return 'Q0'
    elif (feature >Q1 and feature <= Q2):
        return 'Q1'
    elif (feature > Q2 and feature <= Q3):
        return 'Q2'
    elif feature > Q3:
        return 'Q3'


def continuous_to_categorical(cont_blocks):
    
    BP_cat_data = cont_blocks['BP_data'].copy()
    CreatGlucHgHmT_cat_data = cont_blocks['CreatGlucHgHmT_data'].copy()
    HR_RR_cat_data = cont_blocks['HR_RR_data'].copy()
    pH_cat_data = cont_blocks['pH_data'].copy()
    print "BP_cat_data shape = {}".format(BP_cat_data.shape)
    
    
    
    cont_cat_blocks = {
                       'BP_cat_data':BP_cat_data,
                       'CreatGlucHgHmT_cat_data': CreatGlucHgHmT_cat_data,
                       'HR_RR_cat_data' : HR_RR_cat_data, 
                       'pH_cat_data' : pH_cat_data
                      }

    for key in cont_cat_blocks.keys():
        frame = cont_cat_blocks[key]
        frame_stats = frame.describe()
        for col in frame.columns[3:]:
            Q1 = frame_stats[col].loc['25%']
            Q2 = frame_stats[col].loc['50%']
            Q3 = frame_stats[col].loc['75%']
            frame[col] = frame[col].apply(lambda x: quant_cats(x, Q1, Q2, Q3))

        

    return cont_cat_blocks     


def continuous_categorical_to_dummies(dummies, cont_cat_blocks):
    # CONVERT CONTINUOUS/CATEGORICAL DATA TO DUMMIES

    pre_cols = cont_cat_blocks['BP_cat_data'].columns[:1]
    post_cols = cont_cat_blocks['BP_cat_data'].columns[1:]
    print "BPS_cat_data pre columns"
    print(pre_cols)
    
    BP_dummies = cont_cat_blocks['BP_cat_data'][pre_cols].merge(
                      pd.get_dummies(cont_cat_blocks['BP_cat_data'][post_cols]), 
                      left_index = True, right_index = True, how = 'left', sort = True)
                      
    pre_cols = cont_cat_blocks['CreatGlucHgHmT_cat_data'].columns[:1]
    post_cols = cont_cat_blocks['CreatGlucHgHmT_cat_data'].columns[1:]
    CreatGlucHgHmT_dummies = cont_cat_blocks['CreatGlucHgHmT_cat_data'][pre_cols].merge(
                    pd.get_dummies(cont_cat_blocks['CreatGlucHgHmT_cat_data'][post_cols]), 
                    left_index = True, right_index = True, how = 'left', sort = True)
    print "CreatGlucHgHmT_cat_data pre columns"
    print(pre_cols)                       
    
    pre_cols = cont_cat_blocks['HR_RR_cat_data'].columns[:1]
    post_cols = cont_cat_blocks['HR_RR_cat_data'].columns[1:]
    HR_RR_dummies = cont_cat_blocks['HR_RR_cat_data'][pre_cols].merge(pd.get_dummies(
                            cont_cat_blocks['HR_RR_cat_data'][post_cols]), left_index = True, 
                            right_index = True, how = 'left', sort = True)
    print "HR_RR_cat_data"
    print(pre_cols)  
    
    
    pre_cols = cont_cat_blocks['pH_cat_data'].columns[:1]
    post_cols = cont_cat_blocks['pH_cat_data'].columns[1:]
    pH_dummies = cont_cat_blocks['pH_cat_data'][pre_cols].merge(pd.get_dummies(
                            cont_cat_blocks['pH_cat_data'][post_cols]), left_index = True, 
                            right_index = True, how = 'left', sort = True)
    
    print "pH_cat_data"
    print(pre_cols)  
    
    cont_dummy_dict = {'BP_dummies': BP_dummies, 
                       'CreatGlucHgHmT_dummies' : CreatGlucHgHmT_dummies, 
                       'HR_RR_dummies' : HR_RR_dummies, 
                       'pH_dummies' : pH_dummies
                       }
    
    for frame in cont_dummy_dict.keys():
        cont_dummy_dict[frame].set_index('icustay_id', inplace = True)
        cont_dummy_dict[frame].drop('subject_id', inplace = True, axis = 1)
   
    
    return cont_dummy_dict


def select_features(features_dict): 
    # CREATGLUC ETC HAS ONLY 874 SAMPLES AND SO WON'T BE HELPFUL. 

    root = '../data/features/'

    for name, frame in features_dict.iteritems():#frame = cat_dummy_frames[0]
        X_continuous = frame[frame.columns[1:]]
        y = frame['hospital_expire_flag']
        #display(X_continuous.shape)
        #display(y.shape)
        # ONLY PASSING FRAMES W/ > 5000 ICUSTAYS
        if y.shape[0] > 5000:
        
            # SELECT K BEST FEATURES BASED ON CHI2 SCORES
            selector = SelectKBest(score_func = chi2, k = 'all')
            selector.fit(X_continuous, y)
            p_vals = pd.Series(selector.pvalues_, name = 'p_values', index = X_continuous.columns)
            scores = pd.Series(selector.scores_, name = 'scores', index = X_continuous.columns)
            cont_features_df = pd.concat([p_vals, scores], axis = 1)
            cont_features_df.sort_values(by ='scores', ascending = False, inplace = True)
            best_features = frame[cont_features_df[cont_features_df.p_values < .001].index]
            frame = pd.DataFrame(y).merge(best_features, left_index = True, right_index = True, 
                           how = 'left', sort = True)
            print "{}     {}".format(name, frame.shape)
            frame.to_csv(root + name + '.csv')
            cont_features_df[cont_features_df.p_values < .001].to_csv(root + name + 'Scores.csv')




# IMPORT AND REORGANIZE DATA
print "********************************************************************************"
print "************************ Processing Chart Events Data **************************"
print "********************************************************************************"

data = import_chartevents_data()
# FILTER OUT VARIABLES WITH FEWER THAN 2K SAMPLES AND ORGANIZED 
# DATA BY TYPE, CONTINUOUS, CATEGORICAL, CONSTANT
old_cols_continuous, old_cols_const, old_cols_cat = explore_data(data)




# CALCULATE STATISTICS ON DATA
calc_dicts, const_dict, cat_dict = calculate_stats(data, old_cols_continuous, 
                old_cols_const, old_cols_cat)


'''
frame = calc_dicts.keys()[0]
col = calc_dicts[frame].keys()[0]
col2 = calc_dicts[frame][col].keys()[0]
dummy = calc_dicts[frame][col]
#plot_feature_density(dummy, col, False)
'''
remove_non_variable_features(calc_dicts)
#plot_categorical_features(cat_dict, True)

#plot_continuous_features(frame, True)
#plot_constant_features(const_dict, True)


# MERGING CONTINUOUS AND CONSTANT DATA
data3 = merge_continuous_data(data, calc_dicts, const_dict)
dummies = categorical_to_dummy(data3, cat_dict)
#Cap_dummies, GCS_Total_dummies, GCS_dummies = categorical_affinity_blocks(dummies)
cat_dummy_dict = categorical_affinity_blocks(dummies)


data3 = drop_sparse_data(data3)


cont_blocks = continuous_affinity_blocks(data3)
for key in cont_blocks.keys():
    print "the shape of {} is {}".format(key, cont_blocks[key].shape)
# CONSIDER PACKING THESE UP IN A LIST OR DICT
cont_cat_blocks = continuous_to_categorical(cont_blocks)
for key in cont_cat_blocks.keys():
    print "the shape of {} is {}".format(key, cont_cat_blocks[key].shape)
print "Data3 first 3 columns are..... "
print(data3.columns[:3])


dummies = data3[data3.columns[:3]]
dummies.set_index(['icustay_id'], inplace = True)
cont_dummy_dict = continuous_categorical_to_dummies(dummies, cont_cat_blocks)

for key in cont_dummy_dict.keys():
    print "the shape of {} is {}".format(key, cont_dummy_dict[key].shape)
    
  
dummy_frames = [cont_dummy_dict['BP_dummies'], 
                cont_dummy_dict['CreatGlucHgHmT_dummies'],
                cont_dummy_dict['HR_RR_dummies'], 
                cont_dummy_dict['pH_dummies'], 
                cat_dummy_dict['GCS_Total_dummies'], 
                cat_dummy_dict['GCS_dummies']]
                
dummy_frame_filenames = ['BP_Features', 'CreatGlucHgHmT_Features', 'HrRr_Features', 
                     'GCSTotal_Features','GCS_Features']
features_dict = dict(zip(dummy_frame_filenames, dummy_frames))
for key in features_dict.keys():
    print "the first 10 columns of {} is:".format(key)
    print(features_dict[key][features_dict[key].columns[:6]].head(3))


select_features(features_dict)
print "Chart Events Feature Selection Complete"

