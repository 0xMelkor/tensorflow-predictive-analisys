import os


class DataPreprocessor:
    collect_rule = {
        'light': 1,
        'medium': 2,
        'full': 4
    }

    settings = {
        'data_path': os.path.join(os.path.dirname(__file__), '..', 'data'),
    }

    """
    This class reads data from 'data' folder and creates a CSV file to feed the DNN
    """

    def __init__(self, data_dir=settings['data_path'], collect_rule='light'):
        """
        :param data_dir: Directory where the training data is stored. Default is the 'data' directory 
        :param collect_rule: Choose how much data we should collect 'light', 'medium', 'full'
        """
        self.data_dir = data_dir
        self.collect_rule = collect_rule
