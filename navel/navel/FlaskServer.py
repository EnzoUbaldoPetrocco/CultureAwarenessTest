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


app = Flask(__name__)


@app.route("/navigation", methods=['GET'])
def navigation_command():
    
    if request.method == 'GET':
            response = make_response(self.command)
            response.headers['Content-Type'] = 'text/plain'
            return response
        

@app.route("/image",  methods=['POST', 'GET'])
def post_image():
    if request.method == 'POST':
        image = request.args.get('image', None)
        shape = request.args.get('shape', None)
        print(f"Server: image is")
        print(image)
        image = base64.b64decode(image)
        image = PImage.frombytes("RGB", shape, image)
        image = image[:, :, ::-1]
        #TODO: publish image
        print("Here")
        plt.imshow(image)
        plt.show()
        print("There")
        #TODO: get model inference response
        self.inf_req.image = Image(image)
        self.inf_req.shape = shape
        self.future = self.inf_cli.call_async(self.inf_req)
        rclpy.spin_until_future_complete(self, self.future)
        res = self.future.result()
        response_json = {'prediction':res.res}
        return response_json
    if request.method == 'GET':
         response_json = {'image': image, 'shape':shape}

     



async def main(args=None):
    asyncio.run(app.run(host="130.251.13.139", port=5000, debug=False))

if __name__ == '__main__':
    main()
