#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
from pynput import keyboard
from std_msgs.msg import String

class KeyboardCommands(Node):
    def __init__(self):
        super().__init__("KeyboardCommands")
        self.publisher_ = self.create_publisher(String, 'keyboard_commands', 10)


    def on_press(self, key :keyboard.Key):
        try:
            if key.name in ['up', 'down', 'left', 'right']:
                print('Command key {0} pressed'.format(
                    key.name))
                msg = String()
                msg.data = key.name
                self.publisher_.publish(msg)
        except AttributeError:
            print('special key {0} pressed'.format(
                key))
            msg = String()
            msg.data = "error"
            self.publisher_.publish(msg)


    def on_release(self,key:keyboard.Key):
        try:
            print('special key {0} released'.format(
                key))
            msg = String()
            msg.data = "stop"
            self.publisher_.publish(msg)
        except AttributeError:
            print('special key {0} released'.format(
                key))
            msg = String()
            msg.data = "error"
            self.publisher_.publish(msg)

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
