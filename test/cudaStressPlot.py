import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
# --- IMPORTANT: REPLACE WITH YOUR ACTUAL DATA ---
# Each list below should contain the inference times (in ms) for
# the corresponding model and stress condition.
# The number of data points in each list can vary.

# Example Data Structure (You will replace these with your actual data)
# For plot (a)

#Golden run
for file in os.listdir("golden_run/"):
    path_file = "golden_run/"+file

    with open(path_file, "r") as opened:
        if str(file).startswith("inception-v4"):
            inception_v4_no_stress= []
            for y in opened.readlines():
                inception_v4_no_stress.append(float(y.strip())*1000)
        elif str(file).startswith("ssd-mobilenet-v1"):
            ssd_mobilenet_v1_no_stress= []
            for y in opened.readlines():
                ssd_mobilenet_v1_no_stress.append(float(y.strip())*1000)

#CPU RUN
for file in os.listdir("GPU_run/"):
    path_file = "GPU_run/"+file


    with open(path_file, "r") as opened:
        if str(file).startswith("inception-v4"):

            if str(file).endswith("gpurun800.txt"):

                inception_v4_gpu800= []
                for y in opened.readlines():
                    inception_v4_gpu800.append(float(y.strip())*1000)

            elif str(file).endswith("gpurun1600.txt"):
                inception_v4_gpu1600= []
                for y in opened.readlines():
                    inception_v4_gpu1600.append(float(y.strip())*1000)

            elif str(file).endswith("gpurun2400.txt"):
                inception_v4_gpu2400= []
                for y in opened.readlines():
                    inception_v4_gpu2400.append(float(y.strip())*1000)

            elif str(file).endswith("gpurun3200.txt"):
                inception_v4_gpu3200= []
                for y in opened.readlines():
                    inception_v4_gpu3200.append(float(y.strip())*1000)

        elif str(file).startswith("ssd-mobilenet-v1"):
            if str(file).endswith("gpurun800.txt"):

                ssd_mobilenet_v1_gpu800= []
                for y in opened.readlines():
                    ssd_mobilenet_v1_gpu800.append(float(y.strip())*1000)

            elif str(file).endswith("gpurun1600.txt"):
                ssd_mobilenet_v1_gpu1600= []
                for y in opened.readlines():
                    ssd_mobilenet_v1_gpu1600.append(float(y.strip())*1000)

            elif str(file).endswith("gpurun2400.txt"):
                ssd_mobilenet_v1_gpu2400= []
                for y in opened.readlines():
                    ssd_mobilenet_v1_gpu2400.append(float(y.strip())*1000)

            elif str(file).endswith("gpurun3200.txt"):
                ssd_mobilenet_v1_gpu3200= []
                for y in opened.readlines():
                    ssd_mobilenet_v1_gpu3200.append(float(y.strip())*1000)

# --- Plot (a) Data Preparation ---
data_a = []
labels_a = []
models_a = []

stress_conditions = ["no stress", "gpu800", "gpu1600", "gpu2400", "gpu3200"]


# Inception v4
for i, condition in enumerate(stress_conditions):
    data = locals()[f'inception_v4_{condition.replace(" ", "_")}']
    for val in data:
        data_a.append(val)
        labels_a.append(condition)
        models_a.append("Inception v4")

df_a = pd.DataFrame({
    'Inference time (ms)': data_a,
    'Condition': labels_a,
    'Model': models_a
})

# --- Plot (b) Data Preparation ---
data_b = []
labels_b = []
models_b = []


# SSD MobileNet v1
for i, condition in enumerate(stress_conditions):
    data = locals()[f'ssd_mobilenet_v1_{condition.replace(" ", "_")}']
    for val in data:
        data_b.append(val)
        labels_b.append(condition)
        models_b.append("SSD MobileNet v1")


df_b = pd.DataFrame({
    'Inference time (ms)': data_b,
    'Condition': labels_b,
    'Model': models_b
})


# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True) # Two subplots, sharing x-axis

# Plot (a)
ax = sns.boxplot(ax=axes[0], x='Condition', y='Inference time (ms)', hue='Model', data=df_a,
            palette={'Inception v4': 'skyblue'})
[ax.axvline(x+.5,color='k', alpha=0.5) for x in ax.get_xticks()]
axes[0].set_title('(a) Inception v4')
axes[0].set_yticks(range(5)) # Set y-axis to logarithmic scale as in the image
axes[0].set_ylabel('Inference time (ms)')
axes[0].set_xlabel('') # Remove x-label for the top plot
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].legend(title='Model')
axes[0].set(ylim=(0, 250))
axes[0].set_yticks([0, 50, 100, 150, 200, 250])
# Plot (b)
bx =sns.boxplot(ax=axes[1], x='Condition', y='Inference time (ms)', hue='Model', data=df_b,
            palette={'SSD MobileNet v1': 'orchid'})
[bx.axvline(x+0.5,color='k', alpha=0.5) for x in ax.get_xticks()]
axes[1].set_title('(b) SSD MobileNet v1')
axes[1].set_yticks(range(5)) # Set y-axis to logarithmic scale as in the image
axes[1].set_ylabel('Inference time (ms)')
axes[1].set_xlabel('') # Remove x-label, as it will be rotated
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].legend(title='Model')
axes[1].set(ylim=(0, 250))
axes[1].set_yticks([0, 50, 100, 150, 200, 250])

plt.subplots_adjust()
plt.tight_layout()
plt.show()