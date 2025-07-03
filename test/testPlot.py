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
        elif str(file).startswith("inception-v1"):
            pass
        elif str(file).startswith("ssd-mobilenet-v1"):
            ssd_mobilenet_v1_no_stress= []
            for y in opened.readlines():
                ssd_mobilenet_v1_no_stress.append(float(y.strip())*1000)
        elif str(file).startswith("ssd-mobilenet-v2"):
            ssd_mobilenet_v2_no_stress= []
            for y in opened.readlines():
                ssd_mobilenet_v2_no_stress.append(float(y.strip())*1000)
        elif str(file).startswith("mobilenet-v1"):
            pass
        elif str(file).startswith("mobilenet-v2"):
            pass


#CPU RUN
for file in os.listdir("cpu_run/"):
    path_file = "cpu_run/"+file


    with open(path_file, "r") as opened:
        if str(file).startswith("inception-v4"):
            inception_v4_cpu= []
            for y in opened.readlines():
                inception_v4_cpu.append(float(y.strip())*1000)
        elif str(file).startswith("inception-v1"):
            pass
        elif str(file).startswith("ssd-mobilenet-v1"):
            ssd_mobilenet_v1_cpu= []
            for y in opened.readlines():
                ssd_mobilenet_v1_cpu.append(float(y.strip())*1000)
        elif str(file).startswith("ssd-mobilenet-v2"):
            ssd_mobilenet_v2_cpu= []
            for y in opened.readlines():
                ssd_mobilenet_v2_cpu.append(float(y.strip())*1000)
        elif str(file).startswith("mobilenet-v1"):
            pass
        elif str(file).startswith("mobilenet-v2"):
            pass


#VM RUN
for file in os.listdir("vm_run/"):
    path_file = "vm_run/"+file


    with open(path_file, "r") as opened:
        if str(file).startswith("inception-v4"):
            inception_v4_vm= []
            for y in opened.readlines():
                inception_v4_vm.append(float(y.strip())*1000)
        elif str(file).startswith("inception-v1"):
            pass
        elif str(file).startswith("ssd-mobilenet-v1"):
            ssd_mobilenet_v1_vm= []
            for y in opened.readlines():
                ssd_mobilenet_v1_vm.append(float(y.strip())*1000)
        elif str(file).startswith("ssd-mobilenet-v2"):
            ssd_mobilenet_v2_vm= []
            for y in opened.readlines():
                ssd_mobilenet_v2_vm.append(float(y.strip())*1000)
        elif str(file).startswith("mobilenet-v1"):
            pass
        elif str(file).startswith("mobilenet-v2"):
            pass


inception_v1_no_stress = []
inception_v1_cpu = []
inception_v1_vm = [ ]
inception_v1_memcpy = [ ]
inception_v1_interrupt = [ ]
inception_v1_open = [ ]
inception_v1_fork = [ ]
inception_v1_udp = [ ]


inception_v4_memcpy = [ ]
inception_v4_interrupt = [ ]
inception_v4_open = [ ]
inception_v4_fork = [ ]
inception_v4_udp = [ ]

mobilenet_v1_no_stress = [ ]
mobilenet_v1_cpu = [ ]
mobilenet_v1_vm = [ ]
mobilenet_v1_memcpy = [ ]
mobilenet_v1_interrupt = [ ]
mobilenet_v1_open = [ ]
mobilenet_v1_fork = [ ]
mobilenet_v1_udp = [ ]

# For plot (b) - you'd add similar data for these models
mobilenet_v2_no_stress = [ ]
mobilenet_v2_cpu = [ ]
mobilenet_v2_vm = [ ]
mobilenet_v2_memcpy = [ ]
mobilenet_v2_interrupt = [ ]
mobilenet_v2_open = [ ]
mobilenet_v2_fork = [ ]
mobilenet_v2_udp = [ ]


ssd_mobilenet_v1_memcpy = [ ]
ssd_mobilenet_v1_interrupt = [ ]
ssd_mobilenet_v1_open = [ ]
ssd_mobilenet_v1_fork = [ ]
ssd_mobilenet_v1_udp = [ ]

ssd_mobilenet_v2_memcpy = [ ]
ssd_mobilenet_v2_interrupt = [ ]
ssd_mobilenet_v2_open = [ ]
ssd_mobilenet_v2_fork = [ ]
ssd_mobilenet_v2_udp = [ ]


# --- Plot (a) Data Preparation ---
data_a = []
labels_a = []
models_a = []

stress_conditions = ["no stress", "cpu", "vm", "memcpy", "interrupt", "open", "fork", "udp"]

# Inception v1
for i, condition in enumerate(stress_conditions):
    data = locals()[f'inception_v1_{condition.replace(" ", "_")}']
    for val in data:
        data_a.append(val)
        labels_a.append(condition)
        models_a.append("Inception v1")

# Inception v4
for i, condition in enumerate(stress_conditions):
    data = locals()[f'inception_v4_{condition.replace(" ", "_")}']
    for val in data:
        data_a.append(val)
        labels_a.append(condition)
        models_a.append("Inception v4")

# MobileNet v1
for i, condition in enumerate(stress_conditions):
    data = locals()[f'mobilenet_v1_{condition.replace(" ", "_")}']
    for val in data:
        data_a.append(val)
        labels_a.append(condition)
        models_a.append("MobileNet v1")

df_a = pd.DataFrame({
    'Inference time (ms)': data_a,
    'Condition': labels_a,
    'Model': models_a
})

# --- Plot (b) Data Preparation ---
data_b = []
labels_b = []
models_b = []

# MobileNet v2
for i, condition in enumerate(stress_conditions):
    data = locals()[f'mobilenet_v2_{condition.replace(" ", "_")}']
    for val in data:
        data_b.append(val)
        labels_b.append(condition)
        models_b.append("MobileNet v2")

# SSD MobileNet v1
for i, condition in enumerate(stress_conditions):
    data = locals()[f'ssd_mobilenet_v1_{condition.replace(" ", "_")}']
    for val in data:
        data_b.append(val)
        labels_b.append(condition)
        models_b.append("SSD MobileNet v1")

# SSD MobileNet v2
for i, condition in enumerate(stress_conditions):
    data = locals()[f'ssd_mobilenet_v2_{condition.replace(" ", "_")}']
    for val in data:
        data_b.append(val)
        labels_b.append(condition)
        models_b.append("SSD MobileNet v2")

df_b = pd.DataFrame({
    'Inference time (ms)': data_b,
    'Condition': labels_b,
    'Model': models_b
})

# --- Plotting ---
fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True) # Two subplots, sharing x-axis

# Plot (a)
ax = sns.boxplot(ax=axes[0], x='Condition', y='Inference time (ms)', hue='Model', data=df_a,
            palette={'Inception v1': 'lightgreen', 'Inception v4': 'skyblue', 'MobileNet v1': 'salmon'})
[ax.axvline(x+.5,color='k', alpha=0.5) for x in ax.get_xticks()]
axes[0].set_title('(a) Inception v1, Inception v4, MobileNet v1 tests')
axes[0].set_yscale('log') # Set y-axis to logarithmic scale as in the image
axes[0].set_ylabel('Inference time (ms)')
axes[0].set_xlabel('') # Remove x-label for the top plot
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend(title='Model')


# Plot (b)
bx =sns.boxplot(ax=axes[1], x='Condition', y='Inference time (ms)', hue='Model', data=df_b,
            palette={'MobileNet v2': 'darkorange', 'SSD MobileNet v1': 'orchid', 'SSD MobileNet v2': 'gold'})
[bx.axvline(x+.5,color='k', alpha=0.5) for x in ax.get_xticks()]
axes[1].set_title('(b) MobileNet v2, SSD MobileNet v1, SSD MobileNet v2 tests')
axes[1].set_yscale('log') # Set y-axis to logarithmic scale
axes[1].set_ylabel('Inference time (ms)')
axes[1].set_xlabel('') # Remove x-label, as it will be rotated
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].legend(title='Model')


plt.tight_layout()
plt.show()