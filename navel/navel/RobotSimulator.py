#!/usr/bin/env python3.10
import rclpy
from rclpy.node import Node
import json
from flask import Flask, request, make_response
import requests
import base64
import cv2
import os
from pathlib import Path
import random
import time
from PIL import Image
import socket
import numpy as np
from io import BytesIO

ip = "130.251.13.139"

url = "http://" + ip + ":5000/"

random.seed(time.time())

class RobotSimulator(Node):
    def __init__(self):
        super().__init__("RobotSimulator")
        self.get_logger().info(f"Service available at ip: {ip}")
        self.init_ds()
        
        self.timer = self.create_timer(1, self.send_random_image)

    def init_ds(self):
        rt = "/home/rice/enzo/FINALDS"
        
        lamps_paths = [
            rt + "/lamps/chinese/120/RGB",
            rt + "/lamps/french/120/RGB",
            rt + "/lamps/turkish/120/RGB",
        ]

        carpet_paths = [
            rt + "/carpets_stretched/indian/200/RGB",
            rt + "/carpets_stretched/japanese/200/RGB",
            rt + "/carpets_stretched/scandinavian/200/RGB",
        ]


        self.init_spec_dataset(lamps_paths, 1)
        self.init_spec_dataset(carpet_paths, 0)

    def init_spec_dataset(self, paths, lamp):
        self.dataset = []
        for j, path in enumerate(paths):
            labels = self.get_labels(path)
            imgs_per_culture = []
            for i, label in enumerate(labels):
                images = self.get_images(path + "/" + label)
                X = []
                for k in range(len(images)):
                    X.append([images[k], [j, i]])
                # print(f"Culture is {j}, label is {i}")
                # plt.imshow(images[0])
                # plt.show()
                imgs_per_culture.append(X)
                del images
                del X
            self.dataset.append(imgs_per_culture)

        if lamp:
            self.lamp_ds = self.dataset
        else:
            self.carpet_ds = self.dataset

        self.dataset = None

    def get_labels(self, path):
        """
        get_labels returns a list of the labels in a directory

        :param path: directory in which search of the labels
        :return list of labels
        """
        dir_list = []
        for file in os.listdir(path):
            d = os.path.join(path, file)
            if os.path.isdir(d):
                d = d.split("\\")
                if len(d) == 1:
                    d = d[0].split("/")
                d = d[-1]
                dir_list.append(d)
        return dir_list

    def get_images(self, path, n=1000, rescale=False):
        """
        get_images returns min(n, #images contained in a directory)

        :param path: directory in which search for images
        :param n: maximum number of images

        :return list of images
        """
        images = []
        types = ("*.png", "*.jpg", "*.jpeg")
        paths = []
        for typ in types:
            paths.extend(Path(path).glob(typ))
        paths = paths[0 : min(len(paths), n)]
        for i in paths:
            im = cv2.imread(str(i)) 
            if rescale:
                im = im  /255
            im = im[..., ::-1]
            images.append(im)
        return images

    def send_random_image(self):
        img = self.rnd_get_image()
        img = Image.fromarray(img)
        size = img.size
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img = base64.b64encode(buffered.getvalue()).decode('utf-8')

        #img = base64.b64encode(img)
        msg = {'image': img, 'shape': str(size)}
        print(msg)
        req = json.dumps(msg)
        headers = {'content_type': 'application/json'}
        res = requests.post(url+'/image',data=req, verify=False)
        print(res)
        print(res.json())

    def rnd_get_image(self):
        lamp = random.randint(0,1)
        if lamp:
            img, label =  self.lamp_ds[random.randint(0, len(self.lamp_ds)-1)][random.randint(0, len(self.lamp_ds[0])-1)][random.randint(0, len(self.lamp_ds[0][0])-1)]
        else:
            img, label =  self.carpet_ds[random.randint(0, len(self.carpet_ds)-1)][random.randint(0, len(self.carpet_ds[0])-1)][random.randint(0, len(self.carpet_ds[0][0])-1)]
        return img

def main(args=None):
    rclpy.init(args=args)
    robot = RobotSimulator()
    rclpy.spin(robot)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
