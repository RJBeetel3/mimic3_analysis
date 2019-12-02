import os
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import yaml
import numpy as np


def quant_cats(feature, Q1, Q2, Q3):
    
    if feature <=Q1:
        return 'Q0'
    elif (feature >Q1 and feature <= Q2):
        return 'Q1'
    elif (feature > Q2 and feature <= Q3):
        return 'Q2'
    elif feature > Q3:
        return 'Q3'


def import_demog_data():
    
    print("Importing patient demographic data")  
    ptnt_demog = pd.read_csv('../data/Ptnt_Demog_First24.csv')
    return ptnt_demog
    
    
def convert_datetimes(ptnt_demog2):
    
    dates_and_times = ['dob', 'admittime', 'dischtime', 'intime', 'outtime', 'deathtime']
    for thing in dates_and_times:
        print("converting {}".format(thing))
        new_series = pd.to_datetime(ptnt_demog2.loc[:,thing])
        ptnt_demog2.loc[:,thing] = new_series

    return ptnt_demog2

def calculate_durations(ptnt_demog2):    

    print("Calculating ages, duration of stays")
    # len(pd.date_range()) APPEARS TO TAKE A VERY LONG TIME
    for index, row in ptnt_demog2.iterrows():
        if (pd.notnull(row['intime']) & pd.notnull(row['dob'])):
            age_val = len(pd.date_range(end = row['intime'], start = row['dob'], freq = 'A'))
        else: 
            age_val = np.nan
        if (pd.notnull(row['intime']) & pd.notnull(row['outtime'])):
            icu_stay_val = len(pd.date_range(end = row['outtime'], start = row['intime'], freq = 'H'))
        else: 
            icu_stay_val = np.nan
        if (pd.notnull(row['admittime']) & pd.notnull(row['dischtime'])):
            hosp_stay_val = len(pd.date_range(end = row['dischtime'], start = row['admittime'], freq = 'H'))
        else:
            hosp_stay_val = np.nan
    
        ptnt_demog2.set_value(index, 'age', age_val)
        ptnt_demog2.set_value(index, 'icu_stay', icu_stay_val)
        ptnt_demog2.set_value(index, 'hosp_stay', hosp_stay_val)
    print("Reconfiguring columns")
    cols = list(ptnt_demog2.columns)
    cols.pop(cols.index('icd9_code'))
    cols.pop(cols.index('icd9_code.1'))
    cols.pop(cols.index('short_title'))
    cols.pop(cols.index('intime'))
    cols.pop(cols.index('outtime'))
    cols.pop(cols.index('admittime'))
    cols.pop(cols.index('dischtime'))
    cols.pop(cols.index('seq_num'))
    cols.pop(cols.index('dob'))

    #cols.insert(0, cols.pop(cols.index('icustay_id')))
    cols.insert(0, cols.pop(cols.index('hadm_id')))
    cols.insert(1, cols.pop(cols.index('age')))
    cols.insert(2, cols.pop(cols.index('icu_stay')))
    cols.insert(3, cols.pop(cols.index('hosp_stay')))
    cols.insert(len(cols), cols.pop(cols.index('hospital_expire_flag')))


    ptnt_demog2 = ptnt_demog2[cols].copy()
    print("ptnt_demog2 in function")
    print(ptnt_demog2.columns)
  
    # CODE FOR AGE OUTLIERS 
    print("age stats")
    print(ptnt_demog2['age'].describe())
    print ("replacing age outliers")

    age_replace_vals = list(ptnt_demog2[ptnt_demog2['age'] > 110]['age'].unique())
    ptnt_demog2['age'].replace(age_replace_vals, np.nan, inplace = True)

    return ptnt_demog2
    
    

def create_diagnoses_defs(ptnt_demog2):   
    #phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
    print("creating diagnoses definitions")
    definitions = yaml.load(open('../data/hcup_ccs_2015_definitions.yaml', 'r'))

    diagnoses = ptnt_demog[['hadm_id', 'icd9_code', 'short_title']].copy()

    # create mapping of hcup_ccs_2015_definitions to diagnoses icd9 codes
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])

    print("map created")
    # map hcup_ccs_2015 definitions to icd9 diagnoses codes
    diagnoses['HCUP_CCS_2015'] = diagnoses.icd9_code.apply(lambda c: def_map[c][0] if c in def_map else None)
    diagnoses['USE_IN_BENCHMARK'] = diagnoses.icd9_code.apply(lambda c: int(def_map[c][1]) if c in def_map else None)
    #diagnoses['subject_id'] = diagnoses.index
    #diagnoses.set_index(np.arange(diagnoses.shape[0]), inplace = True)


    # create dataframe from the def_map dict so that we can isolate the 
    # definitions that are used in benchmarking

    def_map_df = pd.DataFrame.from_dict(def_map, orient = 'index')
    def_map_df.columns = ['Diagnoses', 'Benchmark']
    diagnoses_bm = list(def_map_df[def_map_df.Benchmark == True].drop_duplicates('Diagnoses').Diagnoses)
    
    return diagnoses_bm, diagnoses
    
    
def create_diagnoses_df(ptnt_demog2, diagnoses_bm, diagnoses):
    
    icustays = list(ptnt_demog2.index)

    # create dataframe with hcup_ccp diagnoses benchmark categories as columns and
    # icustay_id information as indices. if the diagnosis is present for a given icustay the 
    # value is 1, otherwise 0. 

    diagnoses2 = pd.DataFrame(columns = diagnoses_bm, index = icustays)
    diagnoses2.fillna(0, inplace = True)
    #print "created empty diagnoses dataframe"
    for row in diagnoses.iterrows():
        if row[1]['USE_IN_BENCHMARK'] == 1:
            diagnoses2.loc[row[0]][row[1]['HCUP_CCS_2015']] = 1
    
    #print "filled diagnoses dataframe "   
    ptnt_demog2.drop(['subject_id', 'deathtime', 'hadm_id'], inplace = True, axis = 1)
    cols = list(ptnt_demog2.columns)
    cols.insert(0, cols.pop(cols.index('hospital_expire_flag')))
    ptnt_demog2 = ptnt_demog2[cols]
    return ptnt_demog2, diagnoses2
    

        
def continuous_to_categorical(ptnt_demog2):
    demog_stats = ptnt_demog2[ptnt_demog2.columns[1:4]].dropna().describe()
    print(demog_stats)
    for col in ptnt_demog2.columns[1:4]:
            Q1 = demog_stats[col].loc['25%']
            Q2 = demog_stats[col].loc['50%']
            Q3 = demog_stats[col].loc['75%']
            ptnt_demog2[col] = ptnt_demog2[col].apply(lambda x: quant_cats(x, Q1, Q2, Q3))
    
    

def categorical_to_dummies(ptnt_demog_data):
    dummies = ptnt_demog_data[ptnt_demog_data.columns[:1]]
    for col in ptnt_demog_data.columns[1:]:
        chimp = pd.get_dummies(ptnt_demog_data[col], prefix = col)
        dummies = dummies.merge(chimp, left_index = True, right_index = True, 
                           how = 'left', sort = True)

    ## MERGE DUMMY VARIABLES AND DIAGNOSES

    
    return dummies
    


def write_best_features(dummies):
    
    frame = dummies
    X = frame[frame.columns[1:]]
    y = frame['hospital_expire_flag']

        
    # SELECT K BEST FEATURES BASED ON CHI2 SCORES
    selector = SelectKBest(score_func = chi2, k = 'all')
    selector.fit(X, y)
    p_vals = pd.Series(selector.pvalues_, name = 'p_values', index = X.columns)
    scores = pd.Series(selector.scores_, name = 'scores', index = X.columns)
    features_df = pd.concat([p_vals, scores], axis = 1)
    features_df.sort_values(by ='scores', ascending = False, inplace = True)
    print("Feature scores/p_values in descending/ascending order")
    print(features_df.head(20))

    best_features = frame[features_df[features_df.p_values < .001].index]

    frame = pd.DataFrame(y).merge(best_features, left_index = True, right_index = True, 
                    how = 'left', sort = True)


    print("head of selected feature frame ")
    print(frame.head())
    #code for writing features to file    
    root = '../data/features/'
    name = 'Ptnt_Demog_Features.csv'
    name2 = 'Ptnt_Demog_FeaturesScores.csv'
    frame.to_csv(root + name)
    features_df[features_df.p_values < .001].to_csv(root + name2)
    y = pd.DataFrame(y)
    y.to_csv(root + 'outcomes.csv')




print("*******************************************") 
print("patient demographics with unique icu stays")

print("*******************************************")
ptnt_demog = import_demog_data()
ptnt_demog2 = ptnt_demog[~ptnt_demog.index.duplicated(keep='first')].copy()
ptnt_demog2 = convert_datetimes(ptnt_demog2)

print("the shape of ptnt_demog2 = {}".format(ptnt_demog2.shape))
for col in ptnt_demog2.columns:
    print(col)



#print(ptnt_demog.columns)
print("calling calculate_durations")
ptnt_demog2 = calculate_durations(ptnt_demog2)
#print "ptnt_demog2 out of function"




print("calling create_diagnoses_defs")
diagnoses_bm, diagnoses = create_diagnoses_defs(ptnt_demog2)
#print(diagnoses_bm)
#print(diagnoses.head())
print("calling create_diagnoses_df")
ptnt_demog_data, diagnoses2 = create_diagnoses_df(ptnt_demog2, diagnoses_bm, diagnoses)
#print(ptnt_demog_data.head())
print("Calling continuous to categorical conversion")
continuous_to_categorical(ptnt_demog_data)
#print(ptnt_demog_data.head())
print("Calling categorical to dummy variables")
dummies = categorical_to_dummies(ptnt_demog_data)
dummies = dummies.merge(diagnoses2, left_index = True, right_index = True, 
                           how = 'left')
print(dummies.head())

write_best_features(dummies)
print("Patient demographic pre-processing and feature selection complete")


