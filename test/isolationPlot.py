import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# --- Dati aggregati ---
data_a = []
labels_a = []
models_a = []

# --- Lettura dati ---

# Golden Run
for file in os.listdir("golden_run/"):
    if file.startswith("ssd-mobilenet-v1"):
        with open(os.path.join("golden_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("Golden Run")
                models_a.append("SSD-MobileNet-V1 - Golden Run")


# Golden with Isolation
for file in os.listdir("isolation_run/"):
    if file.endswith("goldenisolationrun.txt"):
        with open(os.path.join("isolation_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("Golden Run Isolation on CPU 1")
                models_a.append("SSD-MobileNet-V1 - Golden Run Isolation on CPU 1")


# Golden with JetsonClock
for file in os.listdir("clocks_run/"):
    if file.startswith("ssd-mobilenet-v1"):
        with open(os.path.join("clocks_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("Golden Run with Jetson Clock")
                models_a.append("SSD-MobileNet-V1 - Golden Run with Jetson Clock")


# Golden with JetsonClock and isolation on CPU!
for file in os.listdir("isolation_run/"):
    if file.endswith("goldenisolationclocksrun.txt"):
        with open(os.path.join("isolation_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("Golden Run with Isolation on CPU 1 and Jetson Clock")
                models_a.append("SSD-MobileNet-V1 - Golden Run with Isolation on CPU 1 and Jetson Clock")

# --- Costruzione DataFrame ---
df_a = pd.DataFrame({
    'Inference time (ms)': data_a,
    'Condition': labels_a,
    'Model': models_a
})

# --- Plotting ---
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Condition', y='Inference time (ms)', hue='Model', data=df_a,
                 palette='Set2')

ax.set_title('Inference Time for SSD-MobileNet-V1 Across Different Conditions', fontsize=14)
ax.set_ylabel('Inference time (ms)')
ax.set_xlabel('')
ax.set_ylim(30, 70)
ax.set_yticks([30, 40, 50, 60, 70])
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Model')

plt.tight_layout()
plt.show()
