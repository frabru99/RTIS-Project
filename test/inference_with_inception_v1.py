import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import argparse
import os
import time
from time import perf_counter


#SCRIPT SOLO PER Inception v1

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("testtype", type=str, help="type of the test executed")
args = parser.parse_args()


# Carica il modello Inception v1 da TensorFlow Hub
model_url = "https://tfhub.dev/google/imagenet/inception_v1/classification/5"
model = hub.load(model_url)

# Carica le etichette di ImageNet
labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
)

with open(labels_path, "r") as f:
    labels = f.read().splitlines()


times=[]

for image in os.listdir(args.filename):
    img = Image.open(BytesIO()).resize((224, 224))

    # Preprocessamento: normalizza e aggiunge batch dimensione
    img = np.array(img) / 255.0
    img = image.astype(np.float32)
    img = tf.expand_dims(img, axis=0)

    t1= perf_counter()
    logits = model(image)
    prediction = tf.nn.softmax(logits)
    t2 = perf_counter()

    times.append(t2-t1)

    # Stampa la classe predetta
    predicted_class = tf.argmax(prediction[0]).numpy()
    print(f"Classe predetta: {labels[predicted_class]} (probabilità: {prediction[0][predicted_class].numpy():.2f})")

avg_times = sum(times)/len(times) #process the average time of object detection

print("Average Time of test: ", avg_times)

with open(args.network + "_result_" + args.testtype + ".txt", "a") as file:
     file.write(str(avg_times) + "\n")