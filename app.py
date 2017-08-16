# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn
from tflearn.data_utils import load_csv

from data_loader.preprocess import DataPreprocessor


class RUL:
    def __init__(self, data_processor):
        self.data_proc = data_processor

    def preprocess_data(self):
        # Generate csv files from dataset
        self.data_proc.preprocess(force=False)

    def load_data(self, dataset_index, skip_columns=None):
        """
        Loads training set by index. There are 4 training documents
        :param dataset_index: [0-3] index of the document to load
        :param skip_columns: list of columns to be removed from the dataset
        :return: for the selected training set returns 'data' (even skipped) and 'labels' 
        """
        if skip_columns is None:
            skip_columns = []
        training_file, test_file = self.data_proc.get_files_at_index(dataset_index)
        # Load CSV file, indicate that the last column represents labels
        # target_column indicates the labels column
        data, labels = load_csv(training_file, categorical_labels=False)

        if skip_columns:
            for col in skip_columns:
                RUL._skip_column(data, col)

        np_data = np.array(data, dtype=np.double)
        np_labels = np.array(labels, dtype=np.int32)
        np_labels = np_labels[:, np.newaxis] # add one axis

        return np_data, np_labels

    @staticmethod
    def build_neural_network(input_len):
        net = tflearn.input_data(shape=[None, input_len])
        net = tflearn.fully_connected(incoming=net, n_units=32, activation='tanh')
        net = tflearn.fully_connected(incoming=net, n_units=32, activation='tanh')
        net = tflearn.fully_connected(incoming=net, n_units=1, activation='tanh')
        net = tflearn.regression(net)

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
batch_size = 64
n_epoch = 100

rul = RUL(data_processor=DataPreprocessor())
rul.preprocess_data()
data, labels = rul.load_data(dataset_index=0, skip_columns=[0])

# Define model
net = RUL.build_neural_network(data.shape[1])
model = tflearn.DNN(net)

# add one axis
labels = labels[:, np.newaxis]

# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=n_epoch, batch_size=batch_size, show_metric=True)
