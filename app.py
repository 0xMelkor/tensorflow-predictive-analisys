# -*- coding: utf-8 -*-
from __future__ import print_function
from tflearn.data_utils import load_csv
from data_loader.preprocess import DataPreprocessor
import numpy as np
import tflearn


data_proc = DataPreprocessor()
# Generate csv files from dataset
data_proc.preprocess(force=False)

# TODO: Generalize to support load in range 0-3
training_file, test_file = data_proc.get_files_at_index(0)

# Load CSV file, indicate that the last column represents labels
# target_column indicates the labels column
data, labels = load_csv(training_file, categorical_labels=False)

print(data)

