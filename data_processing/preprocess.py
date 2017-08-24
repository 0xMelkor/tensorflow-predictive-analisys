import collections
import csv
import os
from collections import defaultdict
from os.path import isfile

import matplotlib.pyplot as plt
import numpy as np
from tflearn.data_utils import load_csv

del os.environ['TCL_LIBRARY']


class DataPreprocessor:
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.txt_files = [f for f in os.listdir(self.data_path) if
                          (isfile(os.path.join(self.data_path, f)) and not f.startswith('RUL'))]
        self.train_csv_files = []
        self.test_csv_files = []

    def get_data(self, dataset_index, skip_columns=None):
        """
        Loads training set by index. There are 4 training documents
        :param dataset_index: [0-3] index of the document to load
        :param skip_columns: list of columns to be removed from the dataset
        :return: training a tens data and labels
        """
        self._preprocess(force=False)
        if skip_columns is None:
            skip_columns = []

        training_file, test_file = self._get_files_at_index(dataset_index)

        train_x, train_y = load_csv(training_file, categorical_labels=False)
        test_x, test_y = load_csv(test_file, categorical_labels=False)

        if skip_columns:
            for col in skip_columns:
                DataPreprocessor._skip_column(train_x, col)
                DataPreprocessor._skip_column(test_x, col)

        np_train_x = np.array(train_x, dtype=np.float32)
        np_train_y = np.array(train_y, dtype=np.float32)
        np_train_y = np_train_y[:, np.newaxis]  # add one axis

        np_test_x = np.array(test_x, dtype=np.float32)
        np_test_y = np.array(test_y, dtype=np.float32)
        np_test_y = np_test_y[:, np.newaxis]  # add one axis

        return DataPreprocessor._normalize_columns(np_train_x), \
               DataPreprocessor._normalize_columns(np_train_y), \
               DataPreprocessor._normalize_columns(np_test_x), \
               DataPreprocessor._normalize_columns(np_test_y)

    def _preprocess(self, force=False):
        """
        Generates a csv file for each file in the folder
        :param force: Force csv generation even if target folder is not empty 
        """

        # Generate only if force is True or the target directory already contains data
        if force or not self._already_generated():
            for txt_file in self.txt_files:
                self._generate_csv(txt_file)
        self.train_csv_files = [self._get_processed_file_path(f) for f in self.txt_files if f.startswith('train')]
        self.test_csv_files = [self._get_processed_file_path(f) for f in self.txt_files if f.startswith('test')]

    def _get_files_at_index(self, index):
        return self.train_csv_files[index], self.test_csv_files[index]

    def _generate_csv(self, fname):
        """
        Creates a csv file from txt file
        :param fname: source .txt file (i.e. train_FD002.txt) 
        """
        txt_path = os.path.join(self.data_path, fname)
        csv_path = self._get_processed_file_path(fname)
        engine_count = DataPreprocessor._get_time_series_for_each_engine(txt_path)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w') as csv_file:
            filewriter = csv.writer(csv_file, delimiter=',')
            filewriter.writerow(['unit', 'time', 'settings1', 'settings2', 'settings3',
                                 'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5', 'sensor6',
                                 'sensor7', 'sensor8', 'sensor9', 'sensor10', 'sensor11', 'sensor12',
                                 'sensor13', 'sensor14', 'sensor15', 'sensor16', 'sensor17', 'sensor18',
                                 'sensor19', 'sensor20', 'sensor21', 'rul'])
            with open(txt_path, 'r') as txt_file:
                lines = txt_file.readlines()
                for line_index in range(len(lines)):
                    columns = lines[line_index].strip().split(' ')
                    unit = columns[0]
                    engine_count[unit] -= 1
                    columns.append(engine_count[unit])
                    filewriter.writerow(columns)

    def _already_generated(self):
        # FIXME: Use a more robust approach
        flist = os.listdir(self._get_processed_files_dir())
        return len(flist) == 8

    def _get_processed_file_path(self, fname):
        return os.path.join(self._get_processed_files_dir(), fname.replace('.txt', '.csv'))

    def _get_processed_files_dir(self):
        return os.path.join(self.data_path, 'processed')

    @staticmethod
    def _normalize_columns(input_array):
        return input_array / input_array.max(axis=0)

    @staticmethod
    def _skip_column(data, col_index):
        """
        Removes column at index from each row in data
        :param data: Input data from csv (side effect)
        :param col_index: index of the column to remove
        """
        for row in data:
            del row[col_index]

    @staticmethod
    def _get_time_series_for_each_engine(fpath):

        with open(fpath) as input_file:
            lines = input_file.readlines()
            engine_count = defaultdict(int)
            for line_index in range(len(lines)):
                engine = lines[line_index].split(' ')[0]
                engine_count[engine] += 1
        return collections.OrderedDict(sorted(engine_count.items()))


class LstmDataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def reshape_array(data, num_steps, input_len):
        """
        Reshapes input array to be fed into lstm cell. It will
        then reshaped into [batch_size, num_steps, input_len].
        To avoid exceptions during reshape last elements of the array are cutoff
        :param data: nparray of shape [n, input_len]
        :param num_steps: number of sequential inputs
        :param input_len: number of entries for each input
        :return: a reshaped nparray
        """
        truncated = LstmDataProcessor._lstm_truncate_sequence(data, num_steps)
        shape_0 = int(len(truncated) / num_steps)
        shape_1 = num_steps
        shape_2 = input_len
        return truncated.reshape([shape_0, shape_1, shape_2])

    @staticmethod
    def _lstm_truncate_sequence(data_, num_steps):
        new_size = int(len(data_) / num_steps) * num_steps
        return data_[0:new_size]


class PlotUtils:
    def __init__(self):
        self.plot_index = 1

    def make_plot(self, predicted_data, real_data, num_points, mode, save_fname):
        """
        Makes a plot for the given input
        :param predicted_data: nparray for predicted data
        :param real_data: nparray for real data (same size of predicted)
        :param num_points: number of points to be showed, less or at most equal to len(predicted)
        :param mode: enum{'save', 'show'}
        :param save_fname: If provided the image will be saved to file. The file name for the image to save
        """
        if num_points:
            predicted_data = predicted_data[0:num_points]
            real_data = real_data[0:num_points]

        x_axis = np.arange(0, len(predicted_data), 1)
        # Build plot
        plt.figure(self.plot_index)
        plt.plot(x_axis, predicted_data, marker='o', linestyle='--', label='predicted')
        plt.plot(x_axis, real_data, marker='x', label='real')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        self.plot_index += 1

        if mode != 'save':
            plt.show()
        else:
            dest_dir = os.path.join(os.path.dirname(__file__), '../plots')
            dest_file = os.path.join(dest_dir, save_fname)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            plt.savefig(dest_file)
