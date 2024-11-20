#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
from pynput import keyboard
from std_msgs.msg import String
import requests

# Flask Server Configuration
FLASK_SERVER_URL = "http://130.251.13.117:5000"
FLASK_SERVER_URL += "/move"

AVAILABLE_KEYS = ['up', 'down', 'left', 'right', 'w', 's', 'a', 'd']

class KeyboardCommands(Node):
    def __init__(self):
        super().__init__("KeyboardCommands")
        self.publisher_ = self.create_publisher(String, 'keyboard_commands', 10)

    def send_command_to_flask(self,direction):
        try:
            response = requests.put(f"{FLASK_SERVER_URL}?direction={direction}", headers={'Content-Type': 'application/json'})
            self.get_logger().info(f"Sent direction {direction} to Flask server: {response.json()}")
        except Exception as e:
            self.get_logger().error(f"Failed to send direction {direction} to Flask server: {e}")



    def on_press(self, key :keyboard.Key):
        try:
            if key.name in AVAILABLE_KEYS:
                print('Command key {0} pressed'.format(
                    key.name))                
                if key.name == 'up':
                    self.send_command_to_flask('front')
                elif key.char == 'down':
                    self.send_command_to_flask('back')
                elif key.char == 'left':
                    self.send_command_to_flask('left')
                elif key.char == 'right':
                    self.send_command_to_flask('right')
        except AttributeError:
            print('special key {0} pressed'.format(
                key))
            self.send_command_to_flask('stop')

    def on_release(self,key:keyboard.Key):
        try:
            print('special key {0} released'.format(
                key))
            self.send_command_to_flask('stop')
        except AttributeError:
            print('special key {0} released'.format(
                key))
            self.send_command_to_flask('stop')

def main(args=None):
    rclpy.init(args=args)
    kc = KeyboardCommands()

    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=kc.on_press,
        on_release=kc.on_release)
    listener.start()

    rclpy.spin(kc)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
