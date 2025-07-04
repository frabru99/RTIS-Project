from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import argparse
import time
from time import perf_counter
import os

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="folder containing the images to process")
parser.add_argument("testtype", type=str, help="type of the test executed")
parser.add_argument("--network", type=str, default="mobilenet_v1", help="model to use: mobilenet_v1, mobilenet_v2")
args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = f"google/{args.network}_1.0_224"
preprocessor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)
model.to(device)

times = []

for image_file in os.listdir(args.filename):
    image_path = os.path.join(args.filename, image_file)
    img = Image.open(image_path).convert("RGB")

    inputs = preprocessor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t1 = perf_counter()
    outputs = model(**inputs)
    t2 = perf_counter()

    times.append(t2 - t1)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

avg_time = sum(times) / len(times)
print("Average Time of test:", avg_time)

with open(args.network + "_result_" + args.testtype + ".txt", "a") as file:
    file.write(str(avg_time) + "\n")
