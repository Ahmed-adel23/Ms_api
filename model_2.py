from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask_restful import Resource, Api
flask_app = Flask(__name__)
api = Api(flask_app)

# Load the trained LSTM model
model = load_model('istm_model.h5')

# Define label encoders
label_encoders = {}

def process(data):
    df = pd.read_csv(data)
    # Prepare the features and target for training
    df = df.drop(columns=['participantIsControl_encoded'])
    # df = to_categorical(df)
    # Prepare the features and target for testing

    # # # Convert input data to float32
    df = df.astype('float32')
    df = df.values.reshape(df.shape[0], df.shape[1], 1)
    return df

class MSPrediction2(Resource):
    def post(self):
        try:
            # Check if a CSV file is included in the request
            if 'file' not in request.files:
                return {'error': 'No file part'}

            file = request.files['file']

            # Check if the file has an allowed extension (e.g., CSV)
            allowed_extensions = {'csv'}
            if (
                '.' not in file.filename
                or file.filename.split('.')[-1].lower() not in allowed_extensions
            ):
                return {'error': 'Invalid file format'}

            # Read the CSV file and process it
            prodata = process(file)
            prediction = model.predict(prodata)
            predictions = (prediction > 0.5).astype(int)
            mresult = max(predictions[0])
            index = np.argmax(predictions[0])
            labels = ["you are not infected with ms", "you are infected with ms"]
            value = labels[index]
            tf.config.run_functions_eagerly(True)
            return {'prediction_results': value}
        except Exception as e:
            return {'error': KeyError}




