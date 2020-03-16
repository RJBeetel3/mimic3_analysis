""" This module tests functions in the patient demographics module including

the importation, preprocessing and selection of features.

"""



import sys
import os
import pandas as pd
from icu_mortality import DATA_DIR
"""import datetime as datetime
import numpy as np
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import OneHotEncoder
import matplotlib
import matplotlib.pyplot as plt
#import psycopg2
from scipy.stats import ks_2samp
import scipy.stats as scats
import visuals as vs
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
"""

#Define exceptions
class PtntDemogError(Exception): pass
class ImportDataError(PtntDemogError): pass
#class NotIntegerError(RomanError): pass
#class InvalidRomanNumeralError(RomanError): pass


def import_data(ptnt_demog_filename = os.path.join(DATA_DIR,'PTNT_DEMOG_FIRST24.csv')):
    """ import raw data from patient demographics database query
    the demographic data is constant across a patients ICU stay so the code  
    takes the first instance of the data and discards the duplicates. 
    
    :param ptnt_demog_filename: 
    :return: 
    """
    # import patient demographic data from .csv file.
    try:
        print(ptnt_demog_filename)
        ptnt_demog_data = pd.read_csv(ptnt_demog_filename)
        ptnt_demog_data = ptnt_demog_data.drop_duplicates(subset='icustay_id')
    except IOError as e:
        raise ImportDataError
        print(e + "\n")

    return ptnt_demog_data


def convert_datetimes(ptnt_demog_data):
    """ convert date and time data to pandas date_time objects """
    dates_and_times = ['dob', 'admittime', 'dischtime', 'intime', 'outtime', 'deathtime']

    # iterate through the column names and convert each date time text value to
    # pandas date-time objects
    for thing in dates_and_times:
        new_series = pd.to_datetime(ptnt_demog_data.loc[:, thing])
        ptnt_demog_data.loc[:, thing] = new_series
    return ptnt_demog_data








"""
if __name__ == "__main__":

    # for debugging
    #sys.argv = ['thisscript', 'nope.csv']

    script_name, ptnt_demog_filename = sys.argv
    import_data(ptnt_demog_filename)

"""