import subprocess
import matplotlib.pyplot as plt
import re
import warnings

warnings.filterwarnings("ignore")

# Ajusta el rango si tu CPU demora mucho (ej: hasta 10 millones)
# Nota: La CPU es lenta, ten paciencia en la ejecución.
valores_n = range(1000000, 16000000, 2000000) 

# Diccionarios para guardar resultados por modo
# 0: GPU Ord, 1: GPU Rand, 2: CPU Ord, 3: CPU Rand
tiempos = {0: [], 1: [], 2: [], 3: []}
bw = {0: [], 1: [], 2: [], 3: []}

# Nombres y colores para graficar
labels = {
    0: 'GPU Ordenada', 
    1: 'GPU Aleatoria', 
    2: 'CPU Ordenada', 
    3: 'CPU Aleatoria'
}
colors = {0: 'g-o', 1: 'r-x', 2: 'b--s', 3: 'm--^'} 
# Verde(GPU-Ord), Rojo(GPU-Rand), Azul(CPU-Ord), Magenta(CPU-Rand)

print(f"--- Iniciando Benchmark Completo (4 Modos) ---")

for n in valores_n:
    print(f"N = {n}...", end=" ", flush=True)
    
    for modo in [0, 1, 2, 3]:
        try:
            # Ejecutar programa C++
            result = subprocess.run(["./prog", str(n), str(modo)], capture_output=True, text=True)
            
            # Buscar tiempo
            match = re.search(r"Tiempo de Kernel:\s+([0-9.]+)", result.stdout)
            
            if match:
                ms = float(match.group(1))
                sec = ms / 1000.0
                
                tiempos[modo].append(ms)
                
                # Calculo Bandwidth (12 bytes por elemento)
                total_bytes = n * 12
                gb_per_sec = (total_bytes / sec) / 1e9 if sec > 0 else 0
                bw[modo].append(gb_per_sec)
            else:
                print(f"[Err M{modo}]", end=" ")
                tiempos[modo].append(None)
                bw[modo].append(None)

        except Exception as e:
            print(f"[Excep M{modo}]", end=" ")
            tiempos[modo].append(None)
            bw[modo].append(None)

    print("Ok")

# --- CÁLCULO DE SPEEDUP ---
# Speedup Ordenado = CPU Ord / GPU Ord
# Speedup Aleatorio = CPU Rand / GPU Rand
speedup_sorted = []
speedup_random = []

for i in range(len(valores_n)):
    # Ordenado
    if tiempos[2][i] and tiempos[0][i] and tiempos[0][i] > 0:
        s_ord = tiempos[2][i] / tiempos[0][i]
        speedup_sorted.append(s_ord)
    else:
        speedup_sorted.append(0)

    # Aleatorio
    if tiempos[3][i] and tiempos[1][i] and tiempos[1][i] > 0:
        s_rand = tiempos[3][i] / tiempos[1][i]
        speedup_random.append(s_rand)
    else:
        speedup_random.append(0)


# --- GRÁFICOS (3 Subplots) ---
plt.figure(figsize=(18, 5)) # Ancho, Alto

# 1. TIEMPOS (Escala Logarítmica para ver CPU y GPU juntos)
plt.subplot(1, 3, 1)
for m in [3, 2, 1, 0]: # Orden inverso para pintar capas
    if any(tiempos[m]): # Solo si hay datos
        plt.plot(valores_n, tiempos[m], colors[m], label=labels[m])
plt.yscale('log') # IMPORTANTE: Log para ver diferencias grandes
plt.title('Tiempo de Ejecución (Log Scale)')
plt.ylabel('Milisegundos (ms)')
plt.xlabel('N')
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# 2. ANCHO DE BANDA
plt.subplot(1, 3, 2)
for m in [0, 1, 2, 3]:
    if any(bw[m]):
        plt.plot(valores_n, bw[m], colors[m], label=labels[m])
plt.axhline(y=40, color='gray', linestyle=':', label='Teórico MX130')
plt.title('Ancho de Banda Efectivo')
plt.ylabel('GB/s')
plt.xlabel('N')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. SPEEDUP (La nueva gráfica solicitada)
plt.subplot(1, 3, 3)
plt.plot(valores_n, speedup_sorted, 'g-o', label='Speedup (Caso Ordenado)', linewidth=2)
plt.plot(valores_n, speedup_random, 'r-x', label='Speedup (Caso Aleatorio)', linewidth=2)
plt.title('Speedup (Cuántas veces más rápido es GPU vs CPU)')
plt.ylabel('Speedup (X veces)')
plt.xlabel('N')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("benchmark_completo_4modos.png")
print("\n¡Listo! Imagen generada: benchmark_completo_4modos.png")
# plt.show()