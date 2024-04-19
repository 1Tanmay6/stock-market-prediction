from flask import Flask, request, send_file
from pipelines.data_extracter import extract_data
from pipelines.model_usage import predict_route_function
from pipelines.data_preprocessor import preprocess_data
import requests
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def extract_route_function():

    url = request.json.get('url')
    file_name = request.json.get('file_name')
    model = request.json.get('model')

    file_name = './data/' + file_name + '.csv'
    try:
        print(url, file_name, model)
        if not url or not file_name:
            return 'URL and file name are required', 400

        extract_data(url, file_name)
        data = preprocess_data(file_name, model)
        predictions = predict_route_function(data, model)

        # Convert to numpy array for easier manipulation
        predictions = np.array(predictions)

        # Create plot
        plt.plot(predictions)
        plt.title('Plot of Given Data')
        plt.xlabel('Index')
        plt.ylabel('Value')

        # Save the plot as a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Close the plot to release memory
        plt.close()

        # Return the plot as an image
        return send_file(buffer, mimetype='image/png')
    except requests.exceptions.RequestException as e:
        return f'Error extracting data: {str(e)}', 500

# @app.route('/predict', methods=['POST'])
# def predict_route_function():
#     return '''
#     The Prediction from the model will be returned here\n
#     Predict route\n
#     '''

if __name__ == '__main__':
    app.run(port=4000, debug=True)