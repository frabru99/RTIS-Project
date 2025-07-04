# Analysis of AI Workloads on Nvidia Jetson Nano

This project analyzes the inference performance of several well-known AI models on the **NVIDIA Jetson Nano** under different stress conditions. The main goal is to assess the inference time of classification and object detection models in both ideal (baseline) and stressed environments.

## Authors

- Antonio Boccarossa – [a.boccarossa@studenti.unina.it](mailto:a.boccarossa@studenti.unina.it)  
- Francesco Brunello – [f.brunello@studenti.unina.it](mailto:f.brunello@studenti.unina.it)

**Course**: Real Time Systems and Industrial Applications  
**Institution**: Università degli Studi di Napoli Federico II  
**Supervisors**: Marcello Cinque, Andrea Marchetta  
**Date**: 04.07.2025

---

## Models Analyzed

- **Classification Models**:
  - Inception-v1
  - Inception-v4
  - MobileNet-v1
  - MobileNet-v2
- **Object Detection Models**:
  - SSD-MobileNet-v1
  - SSD-MobileNet-v2

---

## Device: NVIDIA Jetson Nano

- **GPU**: 128-core Maxwell
- **CPU**: Quad-core ARM Cortex-A57
- **Memory**: 4 GB LPDDR4
- **OS**: JetPack SDK (pre-installed)
- **Storage**: External SSD used for image datasets

The environment was set up by flashing the OS to an external SSD and installing dependencies required for `jetson-inference`.

---

## Setup & Tools

- **Inference Framework**: [`jetson-inference`](https://github.com/dusty-nv/jetson-inference)
- **Stress Testing Tool**: [`stress-ng`](https://wiki.ubuntu.com/Kernel/Reference/stress-ng)
- **Languages**: Python 3
- **Libraries Used**: `jetson.inference`, `jetson.utils`

Scripts were executed both in a clean "Golden Run" and under stress conditions.

---

## Stress Scenarios

Stress tests simulate real-time load using the following stressors:

| Test Type     | Command |
|---------------|---------|
| CPU           | `stress-ng --cpu 4` |
| Virtual Memory| `stress-ng --vm 4 --vm-bytes 2.5G` |
| Mem Copy      | `stress-ng --memcpy 8` |
| Interrupt     | `stress-ng --clock 4 --aio 4 --aio-requests 30` |
| Open          | `stress-ng --open 4` |
| Fork          | `stress-ng --fork 4` |
| UDP           | `stress-ng --udp 4` |

---

## How the Tests Work

Each test involves:

1. Running inference on **108 images** per model.
2. Logging the average inference time for each batch.
3. Repeating the test **30 times per configuration** to build a statistically valid sample.

Each script receives CLI arguments:
- `filename`: Path to image folder
- `testtype`: Name of the stress test (e.g., `golden`, `cpu`)
- `--network`: Name of the neural network model (e.g., `googlenet`, `ssd-mobilenet-v1`)

---

## Sample Command

```bash
python3 classify_test.py ./images golden --network=googlenet
```

```bash
python3 detect_test.py ./images cpu --network=ssd-mobilenet-v2
```

---

## Results

The inference times were measured and logged for each model in each scenario. Results are stored in `.txt` files in the format:

```
<network>_result_<testtype>.txt
```

Each file contains 30 average inference times (one per run), suitable for statistical analysis (mean, variance, etc.).

---

## References

- [Jetson Inference GitHub](https://github.com/dusty-nv/jetson-inference)
- [Inception-v4 Paper](https://arxiv.org/abs/1602.07261)
- [SSD MobileNet Architecture](https://iq.opengenus.org/ssd-mobilenet-v1-architecture/)
- [Jetson Nano Specs](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
