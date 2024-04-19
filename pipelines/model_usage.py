import joblib

class ModelLoader:
    def __init__(self):
        self.linear_regression_model = None
        self.ann_model = None
        self.rnn_model = None
        self.lstm_model = None

    def load_models(self):
        self.linear_regression_model = joblib.load('./models/pkl/lr_model.pkl')
        self.ann_model = joblib.load('./models/pkl/ann_model.pkl')
        self.rnn_model = joblib.load('./models/pkl/rnn_model.pkl')
        self.lstm_model = joblib.load('./models/pkl/lstm_model.pkl')

    def predict_linear_regression(self, input_data):
        if self.linear_regression_model is None:
            raise ValueError("Linear regression model has not been loaded.")
        predictions = self.linear_regression_model.predict(input_data)
        print('Linear Regression Says the trend would be:', predictions)
        return self.linear_regression_model.predict(input_data)

    def predict_ann(self, input_data):
        if self.ann_model is None:
            raise ValueError("ANN model has not been loaded.")
        predictions = self.ann_model.predict(input_data)
        print('ANN Says the trend would be:', predictions)
        return self.ann_model.predict(input_data)

    def predict_rnn(self, input_data):
        if self.rnn_model is None:
            raise ValueError("RNN model has not been loaded.")
        predictions = self.rnn_model.predict(input_data)
        print('RNN Says the trend would be:', predictions)
        return self.rnn_model.predict(input_data)

    def predict_lstm(self, input_data):
        if self.lstm_model is None:
            raise ValueError("LSTM model has not been loaded.")
        preadictions = self.lstm_model.predict(input_data)
        print('LSTM Says the trend would be:', preadictions)
        return self.lstm_model.predict(input_data)

def predict_route_function(input_data, model):
    model_loader = ModelLoader()
    model_loader.load_models()
    if model == '0':
        return model_loader.predict_linear_regression(input_data)
    elif model == '1':
        return model_loader.predict_ann(input_data)
    elif model == '2':
        return model_loader.predict_rnn(input_data)
    elif model == '3':
        return model_loader.predict_lstm(input_data)
    else:
        raise ValueError("Invalid model name")
    
    # return '''
    # The Prediction from the model will be returned here\n
    # Predict route\n
    # '''
    
if __name__ == '__main__':
    model_loader = ModelLoader()
    model_loader.load_models()
    print(model_loader.predict_linear_regression([[1, 2, 3, 4]]))
    print(model_loader.predict_ann([[1, 2, 3, 4]]))
    print(model_loader.predict_rnn([[1, 2, 3, 4, 1, 2, 3, 4,1, 2, 3, 4,1, 2, 3, 4,1, 2, 3, 4,1, 2, 3, 4,1]]))
    print(model_loader.predict_lstm([[1, 2, 3, 4, 1, 2, 3, 4,1, 2, 3, 4,1, 2, 3, 4,1, 2, 3, 4,1, 2, 3, 4,1]]))