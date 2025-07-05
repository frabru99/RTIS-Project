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
    if file.startswith("inception-v4"):
        with open(os.path.join("golden_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("Golden Run")
                models_a.append("Inception v4 - Golden Run")

# CPU Stress Run
for file in os.listdir("cpu_run/"):
    if file.startswith("inception-v4"):
        with open(os.path.join("cpu_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("CPU Stress Run")
                models_a.append("Inception v4 - CPU Stress Run" )

# Jetson Clock - Golden
for file in os.listdir("clocks_run/"):
    if file.endswith("goldenclocksrun.txt"):
        with open(os.path.join("clocks_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("Golden Run (Jetson Clock)")
                models_a.append("Inception v4 - Golden Run (Jetson Clock)")

# Jetson Clock - CPU Stress
for file in os.listdir("clocks_run/"):
    if file.endswith("cpuclocksrun.txt"):
        with open(os.path.join("clocks_run", file), "r") as f:
            for line in f:
                data_a.append(float(line.strip()) * 1000)
                labels_a.append("CPU Stress Run (Jetson Clock)")
                models_a.append("Inception v4 - CPU Stress Run (Jetson Clock)")

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

ax.set_title('Inference Time for Inception v4 Across Different Conditions', fontsize=14)
ax.set_ylabel('Inference time (ms)')
ax.set_xlabel('')
ax.set_ylim(90, 115)
ax.set_yticks([90, 95, 100, 105, 110, 115])
ax.tick_params(axis='x', rotation=25)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Model')

plt.tight_layout()
plt.show()
