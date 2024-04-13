from flask import Flask, request
from data_extracter import extract_data
import requests

app = Flask(__name__)

@app.route('/extract', methods=['POST'])
def extract_data():
    url = request.json.get('url')
    file_name = request.json.get('file_name')

    if not url or not file_name:
        return 'URL and file name are required', 400

    try:
        extract_data(url, file_name)
        return 'Data extracted and saved successfully'
    except requests.exceptions.RequestException as e:
        return f'Error extracting data: {str(e)}', 500

if __name__ == '__main__':
    app.run(port=4000, debug=True)