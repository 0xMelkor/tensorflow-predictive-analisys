# -*- coding: utf-8 -*-
from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import tflearn
from tflearn.data_utils import load_csv

from data_loader.preprocess import DataPreprocessor

del os.environ['TCL_LIBRARY']


class RUL:
    def __init__(self, data_processor):
        self.data_proc = data_processor
        self.plot_index = 1

    def load_data(self, dataset_index, skip_columns=None):
        """
        Loads training set by index. There are 4 training documents
        :param dataset_index: [0-3] index of the document to load
        :param skip_columns: list of columns to be removed from the dataset
        :return: training a tens data and labels
        """
        self.data_proc.preprocess(force=False)
        if skip_columns is None:
            skip_columns = []

        training_file, test_file = self.data_proc.get_files_at_index(dataset_index)

        train_x, train_y = load_csv(training_file, categorical_labels=False)
        test_x, test_y = load_csv(test_file, categorical_labels=False)

        if skip_columns:
            for col in skip_columns:
                RUL._skip_column(train_x, col)
                RUL._skip_column(test_x, col)

        np_train_x = np.array(train_x, dtype=np.float32)
        np_train_y = np.array(train_y, dtype=np.float32)
        np_train_y = np_train_y[:, np.newaxis]  # add one axis

        np_test_x = np.array(test_x, dtype=np.float32)
        np_test_y = np.array(test_y, dtype=np.float32)
        np_test_y = np_test_y[:, np.newaxis]  # add one axis

        return RUL.normalize_columns(np_train_x), \
               RUL.normalize_columns(np_train_y), \
               RUL.normalize_columns(np_test_x), \
               RUL.normalize_columns(np_test_y)

    @staticmethod
    def normalize_columns(input_array):
        return input_array / input_array.max(axis=0)

    def make_plot(self, predicted_data, real_data, num_points, mode, save_fname ):
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
        plt.figure(self.plot_index)
        plt.plot(x_axis, predicted_data, marker='o', linestyle='--', label='predicted')
        plt.plot(x_axis, real_data, marker='x', label='real data')

        self.plot_index += 1

        if mode != 'save':
            plt.show()
        else:
            dest_dir = os.path.join(os.path.dirname(__file__), 'plots')
            dest_file = os.path.join(dest_dir,save_fname)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            plt.savefig(dest_file)

    @staticmethod
    def build_neural_network(input_len):
        """
        :param input_len: Number of columns for each row of data
        :return: 
        """
        net = tflearn.input_data(shape=[None, input_len])
        net = tflearn.fully_connected(incoming=net, n_units=input_len, activation='sigmoid')
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(incoming=net, n_units=1, activation='sigmoid')
        net = tflearn.regression(net, optimizer='adam', loss='mean_square',
                                 metric='R2', learning_rate=0.001)

        return net

    @staticmethod
    def _skip_column(data, col_index):
        """
        Removes column at index from each row in data
        :param data: Input data from csv (side effect)
        :param col_index: index of the column to remove
        """
        for row in data:
            del row[col_index]


# params
BATCH_SIZE = 128
N_EPOCHS = 50

rul = RUL(data_processor=DataPreprocessor())
data_train_x, data_train_y, test_x, test_y = rul.load_data(dataset_index=0, skip_columns=[0])

# Define model
net = RUL.build_neural_network(data_train_x.shape[1])
model = tflearn.DNN(net)

# Start training
epoch = 1
for epoch in range(N_EPOCHS):
    model.fit(data_train_x, data_train_y, n_epoch=1, batch_size=BATCH_SIZE,show_metric=True)
    prediction = model.predict(data_train_x)
    rul.make_plot(predicted_data=prediction, real_data=data_train_y,
                  num_points=2000, mode='save', save_fname='epoch_%s' % epoch)

print("Optimization finished!")

