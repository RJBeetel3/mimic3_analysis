import os
import unittest
import pandas
import patient_demographics as pdg
from icu_mortality import DATA_DIR

class ptntDemogImportError(unittest.TestCase):
    """
        expected values
        """
    def setUp(self):

        self.expected_shape = (44152, 20)
        self.expected_cols = ['icustay_id',
                             'hadm_id',
                             'subject_id',
                             'first_careunit',
                             'gender',
                             'marital_status',
                             'ethnicity',
                             'insurance',
                             'admission_type',
                             'admittime',
                             'dischtime',
                             'intime',
                             'outtime',
                             'deathtime',
                             'dob',
                             'hospital_expire_flag',
                             'icd9_code',
                             'icd9_code.1',
                             'short_title',
                             'seq_num']

        self.date_time_cols = ['dob',
                              'admittime',
                              'dischtime',
                              'intime',
                              'outtime',
                              'deathtime']
        #print(os.path.join(DATA_DIR,'PTNT_DEMOG_FIRST24.csv'))
        ptnt_demographic_filename = os.path.join(DATA_DIR,'PTNT_DEMOG_FIRST24.csv')
        self.ptnt_demog_data = pdg.import_data(ptnt_demographic_filename)

    def test_import_data_invalid_file(self):
        """test data import function with and invalid filename"""
        self.assertRaises(pdg.ImportDataError, pdg.import_data, 'file_not_there.csv')

    def test_import_data_valid_file(self):
        """
        test that import returns a dataframe when file exists. the shape of the returned
        frame is checked against what is expected
        """

        self.assertEqual(self.ptnt_demog_data.shape, self.expected_shape)

    def test_import_data_no_file(self):
        """
        test that import returns a dataframe when filename is not provided
        The shape of the returned frame is checked against what is expected
        """
        result = pdg.import_data()
        self.assertEqual(result.shape, self.expected_shape)

    def test_column_names(self):
        """
        test that columns are what we expect
        :return:
        """
        cols = list(self.ptnt_demog_data)
        self.assertEqual(cols, self.expected_cols)

    def test_date_time_conversion(self):
        """
        test that dates and times are converted from text to pandas
        date_time objects
        :return:
        """
        ptnt_demog_data2 = pdg.convert_datetimes(self.ptnt_demog_data)
        for col in self.date_time_cols:
            self.assertTrue((type(ptnt_demog_data2.iloc[0][col]) == pandas._libs.tslibs.timestamps.Timestamp) or
                            (type(ptnt_demog_data2.iloc[0][col]) == pandas.NaT) or
                            (pandas.isna(ptnt_demog_data2.iloc[0][col]))
                            )





if __name__ == "__main__":
	unittest.main()
