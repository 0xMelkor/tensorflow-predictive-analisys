import unittest

from data_loader.preprocess import DataPreprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.dp = DataPreprocessor()
        self.dp.preprocess()

    def test_loaded(self):
        train_files = self.dp.train_csv_files
        test_files = self.dp.test_csv_files
        whole_folder = self.dp.txt_files

        rul_files = [f for f in whole_folder if f.startswith('RUL')]

        self.assertTrue(len(train_files) == 4)
        self.assertTrue(len(test_files) == 4)
        self.assertTrue(len(whole_folder) == 8)
        self.assertTrue(len(rul_files) == 0)

    def test_get_batch_files(self):
        train_file, test_file = self.dp.get_files_at_index(0)
        self.assertTrue('train_FD001.csv' in train_file)
        self.assertTrue('test_FD001.csv' in test_file)
