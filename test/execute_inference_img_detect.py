import os

for i in range (30):
	os.system("python3 inference_img_detection.py ./photos/ goldenrun --network inception-v4")
