import pandas as pd
import yaml


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
    ptnt_demog = pd.DataFrame.from_csv('../data/Ptnt_Demog_First24.csv')
    ptnt_demog2 = ptnt_demog[~ptnt_demog.index.duplicated(keep='first')].copy()
    print "Importing patient demographic data"  
    dates_and_times = ['dob', 'admittime', 'dischtime', 'intime', 'outtime', 'deathtime']
    for thing in dates_and_times:
        new_series = pd.to_datetime(ptnt_demog2.loc[:,thing])
        ptnt_demog2.loc[:,thing] = new_series
    return ptnt_demog2

def calculate_durations(ptnt_demog2):    

    print "Calculating ages, duration of stays"
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
    print "Reconfiguring columns"
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


    ptnt_demog2 = ptnt_demog2[cols] #.copy()
    
    '''
    print "replacing age outliers"
    age_replace_vals = list(ptnt_demog2[ptnt_demog2['age'] > 110]['age'].unique())
    ptnt_demog2['age'].replace(age_replace_vals, np.nan, inplace = True)
    '''

def create_diagnoses_defs(ptnt_demog2):   
    #phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
    print "creating diagnoses definitions"
    definitions = yaml.load(open('../data/hcup_ccs_2015_definitions.yaml', 'r'))

    diagnoses = ptnt_demog2[['hadm_id', 'icd9_code', 'short_title']].copy()

    # create mapping of hcup_ccs_2015_definitions to diagnoses icd9 codes
    def_map = {}
    for dx in definitions:
        for code in definitions[dx]['codes']:
            def_map[code] = (dx, definitions[dx]['use_in_benchmark'])

    print "map created"
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
    print "created empty diagnoses dataframe"
    for row in diagnoses.iterrows():
        if row[1]['USE_IN_BENCHMARK'] == 1:
            diagnoses2.loc[row[0]][row[1]['HCUP_CCS_2015']] = 1
    
    print "filled diagnoses dataframe "   
    ptnt_demog2.drop(['subject_id', 'deathtime', 'hadm_id'], inplace = True, axis = 1)
    cols = list(ptnt_demog2.columns)
    cols.insert(0, cols.pop(cols.index('hospital_expire_flag')))
    ptnt_demog2 = ptnt_demog2[cols]
    return ptnt_demog2
    

    
    
    
    
    
print "patient demographics with unique icu stays"
ptnt_demog2 = import_demog_data()
print(ptnt_demog2.head(5))
print "calling calculate_durations"
calculate_durations(ptnt_demog2)
print(ptnt_demog2.head(5))
print "calling create_diagnoses_defs"
diagnoses_bm, diagnoses = create_diagnoses_defs(ptnt_demog2)
print "calling create_diagnoses_df"
ptnt_demog_data = create_diagnoses_df(ptnt_demog2, diagnoses_bm, diagnoses)

