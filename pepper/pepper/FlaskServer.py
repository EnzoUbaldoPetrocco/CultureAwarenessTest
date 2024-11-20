#!/usr/bin/env python3.10
import base64
import rclpy
from rclpy.node import Node
from flask import Flask, make_response
from flask import request
from std_msgs.msg import String
from PIL import Image as PImage

from robot_interfaces.srv import Inference
from sensor_msgs.msg import Image

from matplotlib import pyplot as plt
import asyncio
import os
from flask_cors import CORS
from flask import jsonify
import socket
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import json

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ip = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

# Mock models (Replace with actual model loading logic)
culturally_aware_model = None #tf.keras.models.load_model('culturally_aware_model.h5')
culturally_unaware_model = None #tf.keras.models.load_model('culturally_unaware_model.h5')
discriminator_model = None #tf.keras.models.load_model('discriminator.h5')
discriminator = False #Using discriminator or nothing
sim_with_models = False
cultural_info = None # Cultural info can be 0,1,2 

if sim_with_models:
        # Select the appropriate model based on cultural information
        if cultural_info:
            model = culturally_aware_model
        else:
            model = culturally_unaware_model
else:
    model = None


#Initialization parameters:
major_culture = "Indian"
img_culture = "Unknown"


# Movement Command received from the Keyboard Node via ROS
current_command = 'stop'

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
res_file = 'results.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/move', methods=['PUT'])
def move():
    global current_command
    direction = request.args.get('direction')
    current_command = direction
    # You could log or process this command further if needed
    print(f"Direction received: {direction}")
    return jsonify({"status": "Command received", "direction": current_command})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image = request.files['image']
    if image.filename == '' or not allowed_file(image.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(image.filename)

    # Save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    image = np.array(data['image'])  # Assuming image is received as a list


    if sim_with_models:
        # Select the appropriate model based on cultural information
        if cultural_info:
            prediction = model.predict(image)
            # Use the discriminator to decide the model to use and apply weighted voting
            if discriminator:
                cultural_probs = discriminator_model.predict(image)
                prediction = np.dot(prediction, cultural_probs)
            else:
                prediction = prediction[cultural_info]
        else:
            prediction = model.predict(image)
    else:
        prediction = np.random.random()
        
    # Return the prediction and the command
    res = {
        "prediction": prediction.tolist(),
        "probability": int(prediction),
        "current_command": current_command
    }
    line = {
        "prediction": prediction.tolist(),
        "probability": int(prediction),
        "filepath": filepath
    }
    with open(res_file, "w") as json_file:
        json.dump(line, json_file)

    return jsonify(res)





def main(args=None): 
    app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
