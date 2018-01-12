import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



def import_labevents_data():
    
    
    data = pd.DataFrame.from_csv('../data/LAB_EVENTS_FIRST24.csv')
    data.loc[:,'charttime']  = pd.to_datetime(data.loc[:,'charttime'])
    data = data.sort_values(['icustay_id', 'charttime'],ascending=True)
    # print "data head:"
    # print(data.head(5))
    '''
    The imported data uses subject_id as the index. The following code moves the 
    subject_id data to a column, creates a proper index and reorganizes the columns 
    to have the lab results grouped together.
    '''
    print "reorganizing data"
    data['subject_id'] = data.index
    data.set_index(np.arange(data.shape[0]), inplace = True)
    cols = list(data.columns)
    cols.insert(0, cols.pop(cols.index('icustay_id')))
    cols.insert(1, cols.pop(cols.index('subject_id')))
    data = data[cols]
    print "reorganized data"
    # print(data.head(5))
    
    # keep the first measurement from each icu-stay 
    # data2 = data.drop_duplicates('icustay_id', keep = 'first')
    # data3 = data.drop_duplicates('subject_id', keep = 'first')


    print "The number of unique ICU stays = {}".format(data.drop_duplicates('icustay_id', keep = 'first').shape[0])
    print "The number of unique patients  = {}".format(data.drop_duplicates('subject_id', keep = 'first').shape[0])

    # display the different measurements captured in the database query
    labels = data.label.unique()
    print "The different measurements captured include:"
    print(labels)
    # print "The number of different measurements captured:"
    # print(len(labels))
    
    return data
    
def remove_sparse_data(data):
    # REMOVE VARIABLES FOR WHICH THERE IS LITTLE DATA / FEW ICUSTAYS FOR WHICH DATA WAS RECORDED
    labels = data.label.unique()
    labels2 = []
    
    # determine the number of samples for each measurement in labels
    # if the measurement has greater than 6k data points, add to labels2
    # essentially removing measurements w/ fewer than 6k data points
    for item in labels:
        
        num_samps = data['icustay_id'][data.label == item].dropna().unique().shape[0]
        #num_measures = data[data.label == item][['icustay_id', 'label']].dropna().groupby('icustay_id').count()
        print "{}    {}".format(item, num_samps) #, num_measures)
        if num_samps > 6000:
            print "adding {}".format(item)
            labels2.append(item)
    labels2.sort(key=str.lower)
    return labels2

# code for calculating and displaying affinity maps. 
# come back to later to clean up
def number_of_samples_per_feature(data, labels2):
    # calculating the number of samples taken in 24 hours for each measurement
    item = labels2[0]

    num_samps_df =  data[data.label == item][['icustay_id', 'label']].dropna().groupby('icustay_id').count()
   
    for item in labels2[1:]:
        #num_samps = data['icustay_id'][data.label == item].dropna().unique().shape[0]
        monkey = data[data.label == item][['icustay_id', 'label']].dropna().groupby('icustay_id').count()
        monkey.columns = [item]
        num_samps_df = num_samps_df.merge(monkey,left_index = True, right_index = True, how = 'left', sort = True) 
        #print "{}    {}".format(item, num_measures) #, num_measures)

    #num_samps_df.drop('label', axis=1, inplace = True)
    return num_samps_df  

# code for displaying affinity maps. not essential but including. 
# 
def affinity_maps(num_samps_df):
    missing = num_samps_df.copy()

    for col in missing.columns:
            missing[col] = missing[col].apply(lambda x: 1 if pd.isnull(x) else 0)
        

    missing = missing.sort_values(by ='Oxygen Saturation', axis = 0, ascending = True)
    #plt.rc('font', size=15)   
    #plt.figure(figsize= (5,8))
    plt.xticks(np.arange(0.5, len(missing.columns), 1), missing.columns)
    plt.xticks(rotation = 30, ha = 'right')
    plt.margins(0.0)
    plt.subplots_adjust(bottom = 0.25)
    print "plotting colormap"
    plt.pcolor(missing)
    #ax.set_ylim([0.0,missing.shape[0]])
    plt.savefig("affinity_plot.png")
    plt.show()
    
    plt.close() 
    
def calculate_stats(data, labels2):
    # height and weight are left out from the calculated measures because there was only one
    # measurement so they are constant.
    
    
    


    # labels2 were sorted alphabetically so we order this list accordingly before zipping
    dict_names = ['Creat','CreatUrine', 'Gluc', 'Hemat', 'Lac', 'LacDehyd', 'O2sat', 'pH', 'WBC']

    # CREATE DICTS OF VARIABLE NAMES WITH MEASUREMENT INDICATOR APPENDED AS KEYS AND 
    # LABELS AS ENTRIES
    first_dict_names = dict(zip([item + '_first' for item in dict_names], labels2))
    mean_dict_names = dict(zip([item + '_mean' for item in dict_names], labels2))
    med_dict_names = dict(zip([item + '_med' for item in dict_names], labels2))
    std_dict_names = dict(zip([item + '_std' for item in dict_names], labels2))
    skew_dict_names = dict(zip([item + '_skew' for item in dict_names], labels2))
    min_dict_names = dict(zip([item + '_min' for item in dict_names], labels2))
    max_dict_names = dict(zip([item + '_max' for item in dict_names], labels2))
    slope_dict_names = dict(zip([item + '_slope' for item in dict_names], labels2))
    delta_dict_names = dict(zip([item + '_delta' for item in dict_names], labels2))
    abnflag_dict_names = dict(zip([item + '_abnflag' for item in dict_names], labels2))

    # CREATE LIST OF NAMES_DICTS FOR EASY TRAVERSAL / ITERATION AND FOR ZIPPING INTO DICTIONARY
    names_list = [first_dict_names, mean_dict_names, med_dict_names, std_dict_names, skew_dict_names, 
                  min_dict_names, max_dict_names, slope_dict_names, delta_dict_names, abnflag_dict_names ]
    # CREATE LIST FOR ZIPPING INTO DICTIONARY THE MEASUREMENT TYPE AND THE CORRESPONDING NAMES_DICT
    calc_list = ['first', 'mean', 'med', 'std', 'skew', 'min', 'max', 'slope', 'delta', 'abnflag']

    # CREATE DICTIONARY WHERE KEY IS THE TYPE OF CALCULATION AND THE VALUE IS THE NAMES_DICT 
    names_dict = dict(zip(calc_list, names_list))


    # CREATE DICTIONARIES IN WHICH TO STORE CALCULATED VALUES
    first_dict = {}
    mean_dict = {}
    med_dict = {}
    std_dict = {}
    skew_dict = {}
    kurt_dict = {}
    min_dict = {}
    max_dict = {}
    slope_dict = {}
    delta_dict = {}
    abnflag_dict = {}
    dict_list = [first_dict, mean_dict, med_dict, std_dict, skew_dict, min_dict, max_dict, slope_dict, delta_dict,
                abnflag_dict]
    calc_dict = dict(zip(calc_list, dict_list))

    # ITERATING THROUGH THE VARIABLES, CALCULATING MEANS, MEDIANS, STD, SKEWNESS, MIN AND MAX'S FOR EACH ITERATION
    # VARIABLES WITH TOO FEW MEASUREMENTS TO CALCULATE THINGS LIKE STD WILL BE AUTOMATICALLY ASSIGNED 'NaN' VALUE
    print "Creating data frames for each summary statistic for each time course variable"
    for calc_key in calc_dict.keys():
        for col_key in names_dict[calc_key].keys(): 
            if calc_key == 'mean':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].mean())
            elif calc_key == 'med':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].median())
            elif calc_key == 'std':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].std())
            elif calc_key == 'max':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].max())
            elif calc_key == 'min':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].min())
            elif calc_key == 'first': 
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].first())
            elif calc_key == 'skew':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].skew())
            elif calc_key == 'delta': 
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].last() -
                                                            data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].first())
            elif calc_key == 'abnflag':
                calc_dict[calc_key][col_key] = pd.DataFrame(data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['flag'].apply(lambda x: int(1) if 'abnormal' in x.values else int(0)))
              
            elif calc_key == 'slope':
                time_last = data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['charttime'].last()
                time_first = data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['charttime'].first()
                val_last = data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].last()
                val_first = data[data.label == names_dict[calc_key][col_key]].groupby('icustay_id')['valuenum'].first()
                calc_dict[calc_key][col_key] = pd.DataFrame((val_last - val_first)/((time_last - time_first)/np.timedelta64(1,'h')))           
        
            
            else: 
                print "need to add code for calculating {}".format(calc_key)
                break
            
            calc_dict[calc_key][col_key].replace([np.inf, -np.inf], np.nan, inplace = True)
            calc_dict[calc_key][col_key].columns = [col_key]
            calc_dict[calc_key][col_key]['hospital_expire_flag'] = data.groupby('icustay_id').hospital_expire_flag.first()
            calc_dict[calc_key][col_key]['gender'] = data.groupby('icustay_id').gender.first()

    print "complete"
    return calc_dict


def plot_features(dummy):
    
    for col in dummy.keys():
    
        col2 = dummy[col].columns[0]
    
    
        gender = ['M', 'F'] 
    
        for gend in gender:
        
            #print gend
            dead = dummy[col][(dummy[col].hospital_expire_flag == 1)&
                                  (dummy[col].gender == gend)]
                              #&(const_dict[col][col2] >20)]
            dead.name = 'Non_Survivors'
            live = dummy[col][(dummy[col].hospital_expire_flag == 0)&
                                  (dummy[col].gender == gend)]
                              #&(const_dict[col][col2] >20)]
            live.name = 'Survivors'
    
    
            maxx = 0.99
            minn = 0.01
    
            live_max = live[col2].dropna().max()#quantile(0.999)
            live_min = live[col2].dropna().min()#quantile(0.001)
            dead_max = dead[col2].dropna().max()#quantile(0.999)
            dead_min = dead[col2].dropna().min()#quantile(0.001)
            maxlim = max(live_max, dead_max)
            minlim = min(live_min, dead_min)
        
            
            plt.subplots(figsize = (10,6))
            live[(live[col2] < live_max) & (live[col2] > live_min)][col2].plot.kde(
                                                                                alpha=1.0,label='Survival')
            dead[(dead[col2] < dead_max) & (dead[col2] > dead_min)][col2].plot.kde(
                                                                                alpha=1.0,label='Non-Survivors')
            
            
            plt.subplots_adjust(left = 0.1, bottom = 0.1, 
                    right = 0.9, top = 0.9)
            plt.xlim(minlim, maxlim)
            plt.title('{} measurement on ICU admission'.format(col) + 
                       'vs ICU mortality by gender = {}\n'.format(gend))
            plt.xlabel(col)

            
            
            # add title, labels etc.
           
            plt.legend(loc="upper left", bbox_to_anchor=(0.75,0.75),fontsize=12)
            save_file_name = "../figures/" + col + "_" + gend + "_" + "plot.png"
            print "saving {}".format(save_file_name)
            plt.savefig(save_file_name)
            plt.close()
        
def remove_outliers(calc_dict):
    names_dict = {}
    suffix = '_outliers'

    # SETTING OUTLIER DATA POINTS TO NAN FOR REMOVAL USING DROPNA()
    for calc in calc_dict.keys():
        frame = calc_dict[calc]
        for col in frame.keys():
        # plot
        # print col
            dummy = frame[col]
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
    

def merge_dataframes(data, calc_dict):
    # MERGING INDIVIDUAL CALCULATED FRAMES INTO A SINGLE DATAFRAMEs
    data2 = data.drop_duplicates('icustay_id', keep = 'first')
    data3 = data2.drop(['label', 'charttime', 'valuenum', 'flag'], axis = 1)
    data3.set_index(['icustay_id'], inplace = True)

    for calc_key in calc_dict.keys():
        print "merging {} dataframe".format(calc_key)
        for col_key in calc_dict[calc_key].keys(): 
            col2 = calc_dict[calc_key][col_key]
            data3 = data3.merge(pd.DataFrame(calc_dict[calc_key][col_key][col_key]), left_index = True, 
                               right_index = True, how = 'left', sort = True)
            newcols = list(data3.columns)
            newcols.pop()
            newcols.append(col_key)
            data3.columns = newcols
    return data3


def drop_features(data3):
    # DROPPING COLUMNS WHERE DATA IS SPARSE AND MISSING DATA DOES NOT CORRELATE WITH OTHER VARIABLES. 
    # THESE DETERMINATIONS WERE MADE THROUGH OBSERVATIONS OF THE AFFINITY MAPS AND DROPNA().SHAPE[0] VALUES ABOVE 

    # REMOVING ALL CREATURINE MEASURES
    drop_cols = []
    for item in data3.columns: 
        if (('CreatUrine'in item) | ('LacDehyd' in item) | ('_skew' in item)):
            drop_cols.append(item)
    #drop_cols

    # DROP THE FOLLOWING MEASURES OF THE FOLLOWING VARIABLES
    #drop_names = ['CreatUrine', 'LacDehyd', 'O2sat', 'Lac']
    drop_names = ['O2sat', 'Lac']
    drop_measures = ['_std', '_slope']
    for name in drop_names:
        for ext in drop_measures:
            drop_cols.append(name + ext)
        
    drop_cols
        
    # we could just return the drop_cols variable and do this outside the function
    #print "data3 shape prior to dropping columns"
    #print(data3.dropna().shape)
    data3.drop(drop_cols, inplace = True, axis = 1)
    

def create_feature_blocks(data3): 

    # TO THIS POINT THE ICUSTAY_ID HAS BEEN USED AS THE INDEX OF THE DATAFRAME. 
    # TO USE THESE METHODS WE CREATE A PROPER INDEX


    # BREAKING UP VARIABLES SO THAT WE CAN DROP NAN VALUES AND STILL HAVE SUFFICIENT SAMPLES 
    # TO TRANSFORM AND DO FEATURE SELECTION / SCORING
    # WILL NEED TO MERGE LATER IN A WAY THAT PROVIDES ADEQUATE SAMPLES

    cols1 = [x for x in data3.columns if (('abnflag' not in x) & (('pH' in x) | ('Lac' in x) | ('O2sat' in x)))]
    cols2 = [x for x in data3.columns if (('abnflag' not in x) & (('Creat' in x) | ('Gluc' in x) | ('Hemat' in x) | ('WBC' in x)))]
    cols3 = [x for x in data3.columns if ('abnflag' in x)]

    header = ['hospital_expire_flag']
    for thing in header:
        cols1.insert(0, thing)
        cols2.insert(0, thing) 
        cols3.insert(0, thing)
  

    #display(cols1)
    data3.replace([np.inf, -np.inf], np.nan, inplace = True)
    pHLacO2Sat_data = data3[cols1].dropna()
    print "pHLacO2Sat_data: Shape = "
    print(pHLacO2Sat_data.shape)                              

    CreatGlucHemWBC_data = data3[cols2].dropna()
    print "CreatGlucHemWBC_data: Shape = "
    print(CreatGlucHemWBC_data.shape)

    AbnFlag_data = data3[cols3].dropna()
    print "AbnFlag_data: Shape = "
    print(AbnFlag_data.shape)

       
    cont_frames = [pHLacO2Sat_data, CreatGlucHemWBC_data]
    cat_frames = [AbnFlag_data]
    return cont_frames, cat_frames

def quant_cats(feature, Q1, Q2, Q3):
    if feature <=Q1:
        return 'Q0'
    elif (feature >Q1 and feature <= Q2):
        return 'Q1'
    elif (feature > Q2 and feature <= Q3):
        return 'Q2'
    elif feature > Q3:
        return 'Q3'

def continuous_to_categorical(cont_frames):

    
    CreatGlucHemWBC_cat_data = cont_frames[1].copy()
    pHLacO2Sat_cat_data = cont_frames[0].copy()

    cont_cat_frames = [CreatGlucHemWBC_cat_data, pHLacO2Sat_cat_data]

    for frame in cont_cat_frames:
        frame_stats = frame.describe()
        for col in frame.columns[1:]:
            Q1 = frame_stats[col].loc['25%']
            Q2 = frame_stats[col].loc['50%']
            Q3 = frame_stats[col].loc['75%']
            frame[col] = frame[col].apply(lambda x: quant_cats(x, Q1, Q2, Q3))

    return cont_cat_frames    


def categorical_to_dummy(cont_cat_frames, AbnFlag_data):
    
    CreatGlucHemWBC_dummies = cont_cat_frames[0][cont_cat_frames[0].columns[:1]].merge(pd.get_dummies(cont_cat_frames[0][cont_cat_frames[0].columns[3:]]), left_index = True, right_index = True, 
                           how = 'left', sort = True)
                       
    pHLacO2Sat_dummies = cont_cat_frames[1][cont_cat_frames[1].columns[:1]].merge(pd.get_dummies(cont_cat_frames[1][cont_cat_frames[1].columns[3:]]), left_index = True, right_index = True, 
                           how = 'left', sort = True)


    dummy_frames = [CreatGlucHemWBC_dummies, pHLacO2Sat_dummies, AbnFlag_data]
    dummy_frame_filenames = ['Lab_CreatGlucHemWBC_Features', 'Lab_pHLacO2Sat_Features', 'Lab_AbnFlag_Features']
    dummy_dict = dict(zip(dummy_frame_filenames, dummy_frames))

    for name, frame in dummy_dict.iteritems():
        print "{}      {}".format(name, frame.shape[0])
    
    return dummy_dict
                       

def select_best_features(dummy_dict):
#select k best features using chi2 score and write those features to file

    for name, frame in dummy_dict.iteritems():
        print "{}      {}".format(name, frame.shape[0])

    # CREATGLUC ETC HAS ONLY 874 SAMPLES AND SO WON'T BE HELPFUL. 

    root = '../data/features/'

    for name, frame in dummy_dict.iteritems():#frame = cat_dummy_frames[0]
        X_continuous = frame[frame.columns[1:]]
        y = frame['hospital_expire_flag']
        #display(X_continuous.shape)
        #display(y.shape)
        # ONLY PASSING FRAMES W/ > 5000 ICUSTAYS
        if y.shape[0] > 3000:
        
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





data = import_labevents_data()
print(data.head())
labels2 = remove_sparse_data(data)
print(labels2)
# code for displaying affinity maps and saving figure to file
#num_samps_df = number_of_samples_per_feature(data, labels2)
#affinity_maps(num_samps_df)
calc_dict = calculate_stats(data, labels2)
print(calc_dict['mean'].keys())
#print "plotting mean values"
#dummy = calc_dict['mean']
#plot_features(dummy)
print "calc_dict dropna shape with outliers= {}".format(calc_dict['mean']['WBC_mean'].dropna().shape)

remove_outliers(calc_dict)
print "calc_dict dropna shape without outliers= {}".format(calc_dict['mean']['WBC_mean'].dropna().shape)
dummy = calc_dict['mean']
plot_features(dummy)
data3 = merge_dataframes(data, calc_dict)
print(data3.columns)
drop_features(data3)
cont_frames, cat_frames = create_feature_blocks(data3)
print(cont_frames[0].columns)
print(cont_frames[1].columns)
print(cat_frames[0].columns)
cont_cat_frames = continuous_to_categorical(cont_frames)
print(cont_cat_frames[0].head())
dummy_dict = categorical_to_dummy(cont_cat_frames, cat_frames[0])
select_best_features(dummy_dict)


