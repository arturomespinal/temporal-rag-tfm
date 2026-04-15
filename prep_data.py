import pandas as pd
import numpy as np
import os

# 1. Cargar el CSV
print("Cargando el CSV...")
csv_path = 'data/ambient_temperature_system_failure.csv'
df = pd.read_csv(csv_path)

# Asumimos que la columna de temperatura es la segunda (índice 1)
# Convertimos los valores a una lista de números
valores = df.iloc[:, 1].values 

# 2. Crear las ventanas temporales (ej. tamaño 96 para la predicción)
window_size = 96
windows = []

print(f"Empaquetando datos en ventanas de {window_size}...")
for i in range(len(valores) - window_size):
    windows.append(valores[i : i + window_size])

# 3. Convertir a Tensor de Numpy [muestras, tamaño_ventana, features]
# Añadimos una dimensión al final porque solo tenemos 1 feature (temperatura)
windows_np = np.array(windows)[..., np.newaxis]

# 4. Guardar como el archivo .npy que espera tu script
salida_path = 'data/dataset_prueba.npy'
np.save(salida_path, windows_np)

print(f"¡Éxito! Archivo guardado en {salida_path} con forma {windows_np.shape}")