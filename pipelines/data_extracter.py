import requests
import csv
import logging
import os
import sys

class DataExtractor:
    def __init__(self, url, output_file):
        self.url = url
        self.output_file = output_file
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler for logging
        log_dir = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + 'Time Series Analysis (Stock Market Prediction)'), 'logger')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'data_extraction.log')
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

    def get_data(self):
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                data = response.json()
                self.logger.info('Data extracted successfully...')
                return data
        except Exception as exception:
            self.logger.warning(f'An error occurred: {exception}')
            return None
        
    def save_data(self, data):
        with open(self.output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ["Date", "Open", "High", "Low", "Close", "Adjusted Close", "Volume", "Dividend Amount", "Split Coefficient"]
            writer.writerow(headers)

            # Loop through the JSON data and write each item to the CSV file
            for date, values in data["Time Series (Daily)"].items():
                row = [date] + list(values.values())
                writer.writerow(row)

    def run(self):
        self.logger.info('Extracting data from API...')
        data = self.get_data()
        self.logger.info('Saving data to CSV file...')
        self.save_data(data)
        self.logger.info(F'Data saved to {self.output_file}')

def extract_data(url, output_file):
    data_extractor = DataExtractor(url, output_file)
    data_extractor.run()

if __name__ == '__main__':
    extract_data()