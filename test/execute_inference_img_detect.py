import os

for i in range (30):
	os.system("python3 inference_img_detection.py ./photos/ cpuclocksrun --network inception-v4")
