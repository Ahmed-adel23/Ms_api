from flask import Flask,  jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import io
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
app = Flask(__name__)
api = Api(app)
# Load the trained GRU model
model = load_model('g_model.h5')
sequence_length = 30
target_variable = 'group'
def process(data):
    df = pd.read_csv(data)
    df=df[df['Initial_Symptom'].isna()==False]
    df=df[df['Schooling'].isna()==False]
    df['Initial_EDSS'].fillna(method='ffill', inplace=True)
    df['Final_EDSS'].fillna(method='ffill', inplace=True)
    df.drop(['Unnamed: 0','Initial_EDSS', 'Final_EDSS'], axis =1, inplace = True)
    df.dropna(inplace= True)
    return df

class MSPrediction(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('file', type=FileStorage, location='files')
            args = parser.parse_args()

            file = args['file']

            if file is None:
                return {'error': 'No file provided'}, 400

            data = process(file)  

            # Rest of your code for predictions
            pred_prob = model.predict(data)
            pred = (pred_prob > 0.5).astype(int)
            prediction = "You are not infected with MS" if pred[0][0] == 0 else "You are infected with MS"
            return {'data': prediction}

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
