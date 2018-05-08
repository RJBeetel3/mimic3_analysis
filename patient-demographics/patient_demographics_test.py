import unittest
import pandas
import patient_demographics as pdg


class ptntDemogImportError(unittest.TestCase):
    """
        expected values
        """
    expected_shape = (44152, 20)
    expected_cols = ['icustay_id',
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

    date_time_cols = ['dob',
                      'admittime',
                      'dischtime',
                      'intime',
                      'outtime',
                      'deathtime']


    def import_data(self):
        """import patient demographics dataframe"""
        result = pdg.import_data('PTNT_DEMOG_FIRST24.csv')
        return result

    def test_import_data_invalid_file(self):
        """test data import function"""
        self.assertRaises(pdg.ImportDataError, pdg.import_data, 'file_not_there.csv')

    def test_import_data_valid_file(self):
        """
        test that import returns a dataframe when file exists. the shape of the returned
        frame is checked against what is expected
        """
        result = self.import_data()
        self.assertEqual(result.shape, self.expected_shape)

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
        result = self.import_data()
        cols = list(result.columns)
        self.assertEqual(cols, self.expected_cols)

    def test_date_time_conversion(self):
        """
        test that dates and times are converted from text to pandas
        date_time objects
        :return:
        """
        result = self.import_data()
        ptnt_demog_data = pdg.convert_datetimes(result)
        for col in self.date_time_cols:
            self.assertTrue((type(ptnt_demog_data.iloc[0][col]) == pandas.tslib.Timestamp) or
                            (type(ptnt_demog_data.iloc[0][col]) == pandas.tslib.NaTType))





if __name__ == "__main__":
	unittest.main()
