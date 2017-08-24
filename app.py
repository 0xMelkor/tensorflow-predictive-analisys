# -*- coding: utf-8 -*-
from __future__ import print_function

import tflearn

from data_processing.preprocess import DataPreprocessor, PlotUtils


class RUL:
    def __init__(self, data_processor):
        self.data_proc = data_processor

    @staticmethod
    def build_mlp_dnn(input_len):
        """
        :param input_len: Number of columns for each row of data
        :return: 
        """
        net = tflearn.input_data(shape=[None, input_len])
        net = tflearn.fully_connected(incoming=net, n_units=input_len, activation='sigmoid')
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(incoming=net, n_units=input_len, activation='sigmoid')
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(incoming=net, n_units=input_len, activation='sigmoid')
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(incoming=net, n_units=input_len, activation='sigmoid')
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(incoming=net, n_units=input_len, activation='sigmoid')
        net = tflearn.batch_normalization(net)
        net = tflearn.fully_connected(incoming=net, n_units=1, activation='sigmoid')
        net = tflearn.regression(net, optimizer='adam', loss='mean_square',
                                 metric='R2', learning_rate=0.001)

        return net


# params
BATCH_SIZE = 128
N_EPOCHS = 20

dp = DataPreprocessor()
plt_util = PlotUtils()

rul = RUL(data_processor=dp)
data_train_x, data_train_y, test_x, test_y = dp.get_data(dataset_index=0, skip_columns=[0])

# Define model
net = RUL.build_mlp_dnn(data_train_x.shape[1])
model = tflearn.DNN(net)

# Start training
epoch = 1
for epoch in range(N_EPOCHS + 1):
    model.fit(data_train_x, data_train_y, n_epoch=1, batch_size=BATCH_SIZE, show_metric=True)

    test_data_prediction = model.predict(test_x)
    plt_util.make_plot(predicted_data=test_data_prediction, real_data=test_y,
                       num_points=2000, mode='save', save_fname='fit_test/epoch_%s' % epoch)

    train_data_prediction = model.predict(data_train_x)
    plt_util.make_plot(predicted_data=train_data_prediction, real_data=data_train_y,
                       num_points=2000, mode='save', save_fname='fit_train/epoch_%s' % epoch)

print("Optimization finished!")
