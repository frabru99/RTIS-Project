import os
import time


for i in range(30):
	os.system("python3 inference_obj_detect.py ./photos/ vmrun --network ssd-mobilenet-v2")
