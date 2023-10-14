from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
from flask_restful import Resource, Api
flask_app = Flask(__name__)
api = Api(flask_app)

# Load your trained ResNet50 model (replace with the actual path to your model)
model = load_model('cnn_model.h5')

# Set the expected input shape for the model
input_shape = (150, 150, 3)

# Function to preprocess an image and make predictions
def preprocess_and_predict(image_bytes):
    try:
        # Convert image bytes to a NumPy array
        nparr = np.frombuffer(image_bytes.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = image / 255.0  # Normalize the image
        prediction = model.predict(np.expand_dims(image, axis=0))
        # Get the class label with the highest probability
        class_label = np.argmax(prediction)

        # Map the class label to the class name using your 'code' dictionary
        class_name = get_code(class_label)

        # Get the prediction confidence (probability) as a Python float
        accuracy = float(np.max(prediction))

        return class_name, accuracy

    except Exception as e:
        return str(e), 500

# Function to map class label to class name
def get_code(n):
    code = {'Control-Axial': 0, 'Control-Sagittal': 1, 'MS-Axial': 2, 'MS-Sagittal': 3}
    for x, y in code.items():
        if n == y:
            return x
def image_to_base64(image):
    # Convert the NumPy array to a PIL Image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    # Save the PIL Image to a BytesIO buffer
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")  # You can choose a different format if needed
    
    # Encode the image as base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
def preprocess_image(image):
    try:
        # Load the image from BytesIO into a PIL Image
        image_pil = Image.open(image)
        
        # Resize the input image to the model's expected size (128x128)
        image_pil = image_pil.resize((128, 128))
        
        # Convert the image to grayscale (1 channel)
        image_pil = image_pil.convert('L')
        
        # Normalize and preprocess the input image
        image_np = np.array(image_pil)
        image_np = image_np / 255.0  # Normalize pixel values to [0, 1]
        
        # Expand dimensions to match the model's input shape
        image_np = np.expand_dims(image_np, axis=-1)
        # Duplicate the single channel to create two channels
        image_np = np.concatenate([image_np, image_np], axis=-1)
        
        # Add any additional preprocessing steps here if needed

        return image_np

    except Exception as e:
        return str(e)
class MSPrediction3(Resource):
    def post(self):
        try:
            # Receive an image from the form submission
            image = request.files['image']

            # Make a prediction
            predicted_class, accuracy = preprocess_and_predict(image)
            image_data = preprocess_image(image)
            image_data = image_to_base64(image_data)
                # Return the predicted class label, accuracy, and the uploaded image as base64
            return {'class_label':predicted_class , 'accuracy':float(accuracy), 'image_data':image_data}

        except Exception as e:
            return str(e)




