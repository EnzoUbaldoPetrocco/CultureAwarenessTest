#!/usr/bin/env python3

import rclpy
import rclpy.exceptions
import rclpy.logging
from std_msgs.msg import String
from pynput import keyboard
from rclpy.node import Node
import requests
import json


# Flask Server Configuration
FLASK_SERVER_URL = "http://130.251.13.117:5000"
FLASK_SERVER_URL += "/move"

class KeyboardCommands(Node):
    def __init__(self):
        super().__init__("KeyboardCommands")

    def send_command_to_flask(self,direction):
        try:
            
            response = requests.put(f"{FLASK_SERVER_URL}?direction={direction}", headers={'Content-Type': 'application/json'})
            self.get_logger().info(f"Sent direction {direction} to Flask server: {response.json()}")
        except Exception as e:
            self.get_logger().error(f"Failed to send direction {direction} to Flask server: {e}")

    # Function to publish keyboard commands
    def on_press(self,key):
        try:
            if key.char == 'w':
                self.send_command_to_flask('front')
            elif key.char == 's':
                self.send_command_to_flask('back')
            elif key.char == 'a':
                self.send_command_to_flask('left')
            elif key.char == 'd':
                self.send_command_to_flask('right')
        except AttributeError:
            pass  # Handle special keys if needed

    # Function to stop the robot when key is released
    def on_release(self,key):
        self.send_command_to_flask('stop')
        if key == keyboard.Key.esc:
            return False
        
def main(args=None):
    try:
        rclpy.init(args=args)
        nd = KeyboardCommands()
        listener = keyboard.Listener(on_press=nd.on_press, on_release=nd.on_release)
        listener.start()
        rclpy.spin(nd)

        
    except rclpy.exceptions.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
