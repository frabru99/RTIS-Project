#!/usr/bin/python3

import jetson.inference
import jetson.utils
import os
from os import listdir
from time import perf_counter, sleep
import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="path of the images to process")
parser.add_argument("testtype", type=str, help="type of the test executed (golden, cpu...)")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v1", help="model to use, can be:  googlenet, resnet-18, ect.")
args = parser.parse_args()


times= [] #list of elapsed inference time (for each image within the folder specified in the cli)

#load the recognition network
net = jetson.inference.detectNet(args.network)

for image in os.listdir(args.filename): #for every image in the directory specified

	# load an image (into shared CPU/GPU memory)
	img = jetson.utils.loadImage(image)

	# classify the image
	t1 = perf_counter()
	detections_list = net.Detect(img)
	t2 = perf_counter()
	times.append(t2-t1)
	
	# find the object des
	for i in range(len(detections_list)):
		print("The class detected within the box is {:s} with confidence {:f} :".format(net.GetClassLabel(detections_list[i].ClassID), detections_list[i].Confidence*100))
avg_times = sum(times)/len(times) #process the average time of object detection

print("Average Time of test: ", avg_times)

with open(args.network + "_result_" + args.testtype + ".txt", "a") as file:
     file.write(str(avg_times) + "\n")
	
