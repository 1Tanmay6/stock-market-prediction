import pandas as pd
import numpy as np
import logging
import os
import sys
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, model):
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler for logging
        log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'Time Series Analysis (Stock Market Prediction)'), 'logger')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'data_preprocessing.log')
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setLevel(logging.INFO)

        # Define log format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Create a stream handler for logging to console (terminal)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)  # Apply the same format to the console output


        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.info('Intitializing Data Preprocessor')
        self.logger.info('Data Preprocessor initialized successfully')
        if model == '-1':
            self.logger.warning('No model selected. Returning the data as it is.')

    def preprocess_linear_regression(self, data):
        try:
            self.logger.info("Starting Preprocessing data for Linear Regression")
            # Step 1: Fixing the date format
            data['Date'] = pd.to_datetime(data['Date'])
            data['year'] = data['Date'].dt.year
            data['month'] = data['Date'].dt.month
            data['day'] = data['Date'].dt.day
            self.logger.info("Fixed the Variables")
            # Step 2: Scaling the data
            scaler = MinMaxScaler((0,1))

            data[[x for x in data.columns if x not in ['Date', 'year', 'month', 'day']]] = scaler.fit_transform(data[[x for x in data.columns if x not in ['Date', 'year', 'month', 'day']]])
            self.logger.info("Scaled the data")
            # Step 3: returing the data
            self.logger.info("Data Preprocessing for Linear Regression completed")
            return data
        except Exception as exception:
            self.logger.critical(f'An error occurred: {exception}')
            return data
        
    def preprocess_ann(self, data):
        self.logger.info('Preprocessing data for ANN')
        self.logger.warning('Redirecting the ANN preprocessing over to Linear Regression Preprocessor')
        return self.preprocess_linear_regression(data)

    def preprocess_rnn(self, data, option='Open', time_step=25):
        self.logger.info("Preprocessing data for RNN")
        try:
            # Step 1: Fixing the date format
            data['Date'] = pd.to_datetime(data['Date'])
            data['year'] = data['Date'].dt.year
            data['month'] = data['Date'].dt.month
            data['day'] = data['Date'].dt.day
            self.logger.info("Fixed the Variables and Updated the date time format")
            # Step 2: Reshaping the required array
            data_option = data[option].values
            data_option = data_option.reshape((-1, 1))
            self.logger.info(f"Reshaped the data for the given option({option}).")
            # Step 3: Scaling the features
            scaler = MinMaxScaler((0, 1))
            data_option_scaled = scaler.fit_transform(data_option)
            self.logger.info("Scaled the Features")
            # Step 4: Creating a data feeding sequence
            data_feed = []
            for data_point in range(time_step, len(data_option)):
                data_feed.append(data_option_scaled[data_point - time_step:data_point].flatten())
            data_feed = np.array(data_feed)
            data_feed = np.reshape(data_feed, (data_feed.shape[0],data_feed.shape[1],1))
            self.logger.info("Created the final data feed sequence.")
            return data_feed
        except Exception as exception:
            self.logger.critical(f'An error occurred: {exception}')
            return data
        
    def preprocess_lstm(self, data, option='Open', time_step=25):
        self.logger.info("Preprocessing data for LSTM")
        self.logger.warning('Redirecting the LSTM preprocessing over to RNN Preprocessor')
        return self.preprocess_rnn(data, option, time_step)

def preprocess_data(data, model='-1', option='Open', time_step=25):
    preprocessor = DataPreprocessor(model)
    if model == '0':
        print("Preprocessing data for Linear Regression")
        return preprocessor.preprocess_linear_regression(data)
    elif model == '1':
        print("Preprocessing data for ANN")
        return preprocessor.preprocess_ann(data)
    elif model == '2':
        print("Preprocessing data for RNN")
        return preprocessor.preprocess_rnn(data, option, time_step)
    elif model == '3':
        print("Preprocessing data for LSTM")
        return preprocessor.preprocess_lstm(data, option, time_step)
    else:
        # self.logger.warninging("No model selected. Returning the data as it is.")
        return data

if __name__ == '__main__':
    data = pd.read_csv('./data/stock_TSCO.csv')
    preprocess_data(data)
    preprocess_data(data, '0')
    preprocess_data(data, '1')
    preprocess_data(data, '2')
    preprocess_data(data, '3')