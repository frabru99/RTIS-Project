#!/usr/bin/python3

import jetson.inference
import jetson.utils
import os
from os import listdir
from time import perf_counter
import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("testtype", type=str, help="type of test executed")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect.")
args = parser.parse_args()

net = jetson.inference.imageNet(args.network)

times = []

for image in os.listdir(args.filename):

        # load an image (into shared CPU/GPU memory)
        img = jetson.utils.loadImage(image)

        #classify images
        t1 = perf_counter()

        class_idx, confidence = net.Classify(img)

        t2 = perf_counter()

        times.append(t2-t1)

        # find the object description
        class_desc = net.GetClassDesc(class_idx)

        # print out the result
        print("image is recognized as '{:s}' (class {:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

avg_time = sum(times)/len(times)

with open(args.network + "_result_"+ args.testtype+".txt", "a") as file:
        file.write(str(avg_time)+"\n")

