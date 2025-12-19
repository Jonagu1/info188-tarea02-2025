import multiprocessing
import re
import subprocess

import matplotlib.pyplot as plt


def speedup_calc(n_values, times):
    speedup = []
    speedup_shared = []

    for i in range(len(n_values)):
        # Speedup = T. CPU / T. GPU
        if times[1][i] and times[2][i] > 0:
            s = times[1][i] / times[2][i]
            speedup.append(s)
        else:
            speedup.append(0)

        # Speedup = T. CPU / T. GPU Shared
        if times[1][i] and times[3][i] > 0:
            s = times[1][i] / times[3][i]
            speedup_shared.append(s)
        else:
            speedup_shared.append(0)

    return (speedup, speedup_shared)


def generate_graphs(n_values, times, speedup, speedup_shared):
    labels = {1: "CPU", 2: "GPU", 3: "GPU Memoria Compartida"}

    colors = {1: "r-x", 2: "b--s", 3: "m--^"}

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)

    for num_alg in [3, 2, 1]:
        if any(times[num_alg]):
            plt.plot(n_values, times[num_alg], colors[num_alg], label=labels[num_alg])

    plt.title("Tiempo de Ejecución")
    plt.ylabel("Milisegundos (ms)")
    plt.xlabel("N")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.subplot(1, 2, 2)

    plt.plot(n_values, speedup, "g-o", label="Speedup (GPU)", linewidth=2)
    plt.plot(
        n_values,
        speedup_shared,
        "r-x",
        label="Speedup (GPU Memoria Compartida)",
        linewidth=2,
    )

    plt.title("Speedup (Cuántas veces más rápido es GPU vs CPU)")
    plt.ylabel("Speedup (X veces)")
    plt.xlabel("N")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_matmul.png")


def main():
    n_values = [256, 512, 1024, 2048, 4096]
    # Usamos la máxima cantidad de hilos posibles de CPU para que sea una comparación justa
    nt = multiprocessing.cpu_count()
    times = {1: [], 2: [], 3: []}

    print("[INFO] Inicializando benchmark MATMUL")

    for n in n_values:
        print(f"\n[INFO] Usando N= {n}...")

        for alg, num in [("CPU", 1), ("GPU", 2), ("GPU Memoria Compartida", 3)]:
            print(f"[INFO] Usando algoritmo {alg}...")

            try:
                result = subprocess.run(
                    ["./prog", str(n), str(nt), str(num)],
                    capture_output=True,
                    text=True,
                )

                match = re.search(r"\[INFO\] Tiempo:\s+([0-9.]+)", result.stdout)

                if match:
                    ms = float(match.group(1))
                    times[num].append(ms)
                else:
                    print(f"[ERROR] ¡No se pudo obtener tiempo usando algoritmo {alg}!")
                    times[num].append(None)
            except Exception as e:
                print(f"[ERROR] ¡Ocurrió una excepción al usar algoritmo {alg}!")
                print(f"[ERROR] Mensaje: {e}")
                times[num].append(None)

    print("\n[INFO] Ha finalizado el benchmark. Generando gráficos...")

    speedup, speedup_shared = speedup_calc(n_values, times)

    generate_graphs(n_values, times, speedup, speedup_shared)

    print("[INFO] Ha finalizado la generación de gráficos.")


if __name__ == "__main__":
    main()
