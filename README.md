# prueba 3:

## Descripción
Tarea de programacion en paralelo usando CPU, GPU (Con memoria compartida y/o tensor cores)

## Archivos
* main.cu: El código principal. La CPU prepara los datos (porque es más fácil generar aleatorios ahí) y la GPU hace el cálculo pesado.
* Makefile: Para compilar rápido usando make. Está configurado con -O3 (optimización) y le tuve que agregar el parametro arch=sm_50 para que funcionara en mi tarjeta MX130 que es un poco antigua
* benchmark.py: Un script en Python que hice para automatizar las pruebas, graficar los tiempos y calcular el ancho de banda.
* resultado_benchsmark_test.png: El gráfico resultante con los experimentos.

## Ejecutar

```
make
./prog <n> <nt> <algoritmo>
```
<algoritmo> es para elegir cual paralelismo usar:
    | 1 = CPU multi threads con OpenMP
    | 2 = GPU con CUDA
    | 3 = GPU con CUDA aprovechando la memoria compartida (__shared__)
    | 4 = GPU con Tensor Cores
    
Para hacer pruebas de estos codigos / algoritmos se creo un script de python que ejecuta nuestro programa con los 4 modos en repetidas veces cada uno para crear graficos, se ejecuta con:
```
python3 benchmark.py
```
con esto se generaran los graficos en un png resultado_benchmark.png

## Detalles

en CPU se mide el tiempo con openmp
en GPU se mide el tiempo con cudaEvent para tener la medida exacta de la gpu sin pasar por un procesamiento cpu


## conclusiones

En conclusión, 
