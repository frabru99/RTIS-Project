from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import torch
import argparse
import time
from time import perf_counter

#Use this script only for mobilenet-v1 and mobilenet-v2

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("testtype", type=str, help="type of the test executed")
parser.add_argument("--network", type=str, default="google/mobilenet_v1_1.0_224", help="model to use: mobilenet_v1, mobilenet_v2")
args = parser.parse_args()

"""
MODEL INIZIALIZATION
"""

#Check if the GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

preprocessor = AutoImageProcessor.from_pretrained("google/"+args.network+"_1.0_224")
model = AutoModelForImageClassification.from_pretrained("google/"+args.network+"_1.0_224")

model.to(device)

#list of time predictions
times = []
for image in os.listdir(args.filename):
    img = Image.open(image.raw)

    inputs = preprocessor(images=img, return_tensors="pt")

    t1 = perf_counter()
    outputs = model(**inputs)
    t2 = perf_counter()

    times.append(t2-t1) #inference time

    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

avg_times = sum(times)/len(times) #process the average time of object detection

print("Average Time of test: ", avg_times)

with open(args.network + "_result_" + args.testtype + ".txt", "a") as file:
     file.write(str(avg_times) + "\n")
	