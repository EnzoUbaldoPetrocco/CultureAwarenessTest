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
ip = "130.251.13.139"
ip = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]



app = Flask(__name__)
CORS(app)

ip = "130.251.13.139"
class FlaskServer(Node):
    def __init__(self):
        super().__init__("FlaskServer")
        self.get_logger().info(f"Service available at ip: {ip}")
        self.k_subscriber_ = self.create_subscription(String, 'keyboard_commands',self.keyboard_commands_clbk, 10)
        self.command = "stop"
        self.inf_cli = self.create_client(Inference, 'inference_image')
        while not self.inf_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.inf_req = Inference.Request()
        
    def keyboard_commands_clbk(self, msg: String):
        self.get_logger().info('Keyboard command data = ' % msg.data)
        self.command = msg.data

    def navigation_command(self):
        @app.route("/navigation", methods=['GET'])
        def navigation_command_fn():
            if request.method == 'GET':
                response = make_response(self.command)
                response.headers['Content-Type'] = 'text/plain'
                return response
        return navigation_command_fn
        
    def post_image(self):
        @app.route('/image',  methods=['POST'])
        def post_image_fn():
            if request.method == 'POST':
                """"self.get_logger().info('Inside Server Baby')
                image = request.get_json()['image']#request.args.get('image', None)
                shape = request.get_json()['shape']#request.args.get('shape', None)
                self.get_logger().info(f"Server: image is")
                self.get_logger().info(image)
                image = base64.b64decode(image)
                image = PImage.frombytes("RGB", shape, image)
                image = image[:, :, ::-1]
                #TODO: publish image
                self.get_logger().info("Here")
                plt.imshow(image)
                plt.show()
                self.get_logger().info("There")
                #TODO: get model inference response
                self.inf_req.image = Image(image)
                self.inf_req.shape = shape
                self.future = self.inf_cli.call_async(self.inf_req)
                rclpy.spin_until_future_complete(self, self.future)
                res = self.future.result()
                response_json = {'prediction':res.res}"""
                response_json = {"prediction":"JAN"}
                return jsonify(response_json), 200
        if request.method == "POST":
            return post_image_fn
    




def main(args=None): 
    #os.environ['FLASK_DEBUG_PIN']="142-139-128"
    rclpy.init(args=args)
    fs = FlaskServer()
    async def run():
        await asyncio.gather(app.run(host="130.251.13.139", port=5000, debug=False), rclpy.spin(fs))
    asyncio.run(run())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
