import collections
import csv
import os
from collections import defaultdict

from os.path import isfile


class DataPreprocessor:
    """
    This class reads data from 'data' folder and creates a CSV file to feed the DNN
    """

    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data')

    def preprocess(self):
        txt_files = [f for f in os.listdir(self.data_path) if isfile(os.path.join(self.data_path, f))]
        for txt_file in txt_files:
            self.generate_csv(txt_file)

    def generate_csv(self, fname):
        """
        Creates a csv file from txt file
        :param fname: source .txt file (i.e. train_FD002.txt) 
        """
        txt_path = os.path.join(self.data_path, fname)
        csv_path = os.path.join(self.data_path, 'processed', fname.replace('.txt', '.csv'))
        engine_count = DataPreprocessor.get_time_series_for_each_engine(txt_path)

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

    @staticmethod
    def get_time_series_for_each_engine(fpath):

        with open(fpath) as input_file:
            lines = input_file.readlines()
            engine_count = defaultdict(int)
            for line_index in range(len(lines)):
                engine = lines[line_index].split(' ')[0]
                engine_count[engine] += 1
        return collections.OrderedDict(sorted(engine_count.items()))