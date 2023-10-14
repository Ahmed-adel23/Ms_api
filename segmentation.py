from flask import Flask, request,  jsonify
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64
from flask_restful import Resource, Api
app = Flask(__name__)
api = Api(app)
def preprocess_image(image):
    # Resize the input image to the model's expected size (128x128)
    image = image.resize((128, 128))
    
    # Convert the image to grayscale (1 channel)
    image = image.convert('L')
    
    # Normalize and preprocess the input image
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    
    # Expand dimensions to match the model's input shape
    image = np.expand_dims(image, axis=-1)
    # Duplicate the single channel to create two channels
    image = np.concatenate([image, image], axis=-1)
    
    # Add any additional preprocessing steps here if needed
    return image

def postprocess_mask(mask):
    # Apply thresholding or other postprocessing to the segmentation mask
    # You may need to adjust this based on your model's output
    threshold = 0.5
    mask = (mask > threshold).astype(np.uint8)
    return mask

def image_to_base64(image):
    # Convert the NumPy array to a PIL Image
    image_pil = Image.fromarray((image * 255).astype(np.uint8))
    
    # Save the PIL Image to a BytesIO buffer
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")  # You can choose a different format if needed
    
    # Encode the image as base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
# Define the dice_coef function
def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

# Define the precision function
def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

# Define the sensitivity function
def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(y_true * tf.round(y_pred))
    possible_positives = tf.reduce_sum(y_true)
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

# Define the specificity function
def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    possible_negatives = tf.reduce_sum(1 - y_true)
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

# Load your trained segmentation model
model = tf.keras.models.load_model('D__segmentaion_model.h5', custom_objects={
    'accuracy': tf.keras.metrics.MeanIoU(num_classes=2),
    'dice_coef': dice_coef,
    'precision': precision,
    'sensitivity': sensitivity,
    'specificity': specificity
}, compile=False)

class MSPrediction4(Resource):
    def post(self):
        if request.method == 'POST':
            # Get the uploaded image from the form
            uploaded_file = request.files['file']

            if uploaded_file.filename != '':
                # Read and preprocess the uploaded image
                image = Image.open(uploaded_file)
                image = preprocess_image(image)

                # Perform segmentation using the model
                segmentation_mask = model.predict(np.expand_dims(image, axis=0))[0]

                # Post-process the segmentation mask (e.g., apply a threshold)
                segmentation_mask = postprocess_mask(segmentation_mask)

                # Convert images to base64 for sending in JSON response
                original_image_base64 = image_to_base64(image)
                mask_image_base64 = image_to_base64(segmentation_mask)

                return {'original_image':original_image_base64, 'mask_image':mask_image_base64}

        return jsonify(error="No file uploaded")
