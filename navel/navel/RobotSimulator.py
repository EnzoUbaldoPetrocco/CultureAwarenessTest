from naoqi import ALProxy
import requests
import json
import qi
import sys

# Pepper Configuration
PEPPER_IP = "192.168.1.2"  # Replace with your Pepper's IP
PEPPER_PORT = 9559

# Flask Server Configuration
FLASK_SERVER_URL = "http://130.251.13.139:5000"  # Replace with your Flask server's IP

motion_proxy = ALProxy("ALMotion", PEPPER_IP, PEPPER_PORT)

movement_map = {
    "left": [-1.0, 0.0, 0.0],
    "right": [1.0, 0.0, 0.0],
    "front": [0.0, 1.0, 0.0],
    "back": [0.0, -1.0, 0.0],
    "stop": [0.0, 0.0, 0.0]
}

def move_pepper(direction):
    x, y, theta = movement_map.get(direction, [0.0, 0.0, 0.0])
    motion_proxy.moveTo(x, y, theta)

def send_image_for_prediction(image, cultural_info=None):
    payload = {'image': image}
    if cultural_info:
        payload['cultural_info'] = cultural_info
    try:
        response = requests.post(f"{FLASK_SERVER_URL}/predict", json=payload)
        data = response.json()
        print(f"Prediction: {data['prediction']}, Probability: {data['probability']}")
        move_pepper(data['current_command'])  # Move Pepper based on current command
    except Exception as e:
        print(f"Failed to send POST request for prediction: {e}")

def main():
    session = qi.Session()
    try:
        session.connect("tcp://" + PEPPER_IP + ":" + str(PEPPER_PORT))
    except RuntimeError:
        print ("Can't connect to Pepper at ip \"" + PEPPER_IP + "\" on port " + str(PEPPER_PORT) +".")
        sys.exit(1)

    # Example usage
    # Replace with actual image data capture
    image = [0] * (128 * 128 * 3)  # Example image data as a flat list
    cultural_info = None  # Replace with actual cultural information if needed
    send_image_for_prediction(image, cultural_info)

if __name__ == "__main__":
    main()
