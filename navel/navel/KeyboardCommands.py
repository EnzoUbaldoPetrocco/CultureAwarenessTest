#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from pynput import keyboard
import requests

# Flask Server Configuration
FLASK_SERVER_URL = "http://your-server-ip:5000/move"

def send_command_to_flask(direction):
    try:
        response = requests.get(f"{FLASK_SERVER_URL}?direction={direction}")
        rospy.loginfo(f"Sent direction {direction} to Flask server: {response.json()}")
    except Exception as e:
        rospy.logerr(f"Failed to send direction {direction} to Flask server: {e}")

# Function to publish keyboard commands
def on_press(key):
    try:
        if key.char == 'w':
            send_command_to_flask('front')
        elif key.char == 's':
            send_command_to_flask('back')
        elif key.char == 'a':
            send_command_to_flask('left')
        elif key.char == 'd':
            send_command_to_flask('right')
    except AttributeError:
        pass  # Handle special keys if needed

# Function to stop the robot when key is released
def on_release(key):
    send_command_to_flask('stop')
    if key == keyboard.Key.esc:
        return False

def keyboard_node():
    rospy.init_node('keyboard_node', anonymous=True)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    rospy.spin()

if __name__ == '__main__':
    try:
        keyboard_node()
    except rospy.ROSInterruptException:
        pass
