from model import MSPrediction
from model_2 import MSPrediction2
from classfication import MSPrediction3
from segmentation import MSPrediction4
from flask import Flask
from flask_restful import Api
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/MSPrediction": {"origins": "*"}})
CORS(app, resources={r"/MSPrediction2": {"origins": "*"}})
CORS(app, resources={r"/MSPrediction3": {"origins": "*"}})
CORS(app, resources={r"/MSPrediction4": {"origins": "*"}})
api = Api(app)

api.add_resource(MSPrediction , '/MSPrediction')
api.add_resource(MSPrediction2, '/MSPrediction2')
api.add_resource(MSPrediction3, '/MSPrediction3')        
api.add_resource(MSPrediction4, '/MSPrediction4')

if __name__ == '__main__':
    app.run(debug=False , host='0.0.0.0')