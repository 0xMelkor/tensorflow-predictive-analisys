# -*- coding: utf-8 -*-
from __future__ import print_function

from tflearn.data_utils import load_csv


class RUL:
    def __init__(self, data_processor):
        self.data_proc = data_processor

    def preprocess_data(self):
        # Generate csv files from dataset
        self.data_proc.preprocess(force=False)

    def load_data(self, index):
        training_file, test_file = self.data_proc.get_files_at_index(index)
        # Load CSV file, indicate that the last column represents labels
        # target_column indicates the labels column
        data, labels = load_csv(training_file, categorical_labels=False)
        return data, labels

    @staticmethod
    def _skip_column(data, col_index):
        """
        Removes column at index from each row in data
        :param data: Input data from csv (side effect)
        :param col_index: index of the column to remove
        """
        for row in data:
            del row[col_index]
