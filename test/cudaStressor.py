import torch
import time
from time import perf_counter

# Imposta il dispositivo: usa la GPU se disponibile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Scegli la taglia: 'small', 'medium', 'large'
matrix_size_choice = 'medium'

# Mappa delle dimensioni
size_map = {
    'small': 512,
    'medium': 2048,
    'large': 8192
}

# Imposta N in base alla scelta
N = size_map.get(matrix_size_choice, 512)
print(f"Matrix size: {N}x{N}")

# Genera matrici casuali
A = torch.randn((N, N), dtype=torch.float32, device=device)
B = torch.randn((N, N), dtype=torch.float32, device=device)

# Calcolo
start = time.perf_counter()

C = torch.matmul(A, B)
D = torch.sin(C) + torch.log(torch.abs(C) + 1e-5)
E = torch.matmul(D, A.T)
F = torch.relu(E) ** 2.5
result = torch.sum(F)

# Sincronizza se su GPU
torch.cuda.synchronize() if device.type == "cuda" else None
end = time.perf_counter()

print(f"Computation complete. Result: {result.item():.4e}")
print(f"Elapsed time: {end - start:.2f} seconds")