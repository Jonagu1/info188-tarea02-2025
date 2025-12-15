# prueba 3:

## Descripción
El kernel que hice resuelve la operación A[B[i]] += tid, donde B es como un "hash"

## Archivos
* main.cu: El código principal. La CPU prepara los datos (porque es más fácil generar aleatorios ahí) y la GPU hace el cálculo pesado.
* Makefile: Para compilar rápido usando make. Está configurado con -O3 (optimización) y le tuve que agregar el parametro arch=sm_50 para que funcionara en mi tarjeta MX130 que es un poco antigua
* benchmark.py: Un script en Python que hice para automatizar las pruebas, graficar los tiempos y calcular el ancho de banda.
* resultado_benchsmark_test.png: El gráfico resultante con mis experimentos.

## Ejecutar

```
make
./prog <N> <modo>
```
Se recomienda ejecutar una vez antes de probar con el benchmark para que el prog se "adapte"

Luego para hacer las pruebas de 10 iteraciones con modo =0 y modo = 1 se usa el script de python

```
python3 benchmark.py
```
con esto se generaran los graficos en un png resultado_benchmrk.png

## Detalles

No usé Memoria Compartida (__shared__) porque en este algoritmo cada dato se usa una sola vez. Copiar los datos a la memoria compartida hubiera sido trabajar de más sin ganar rendimiento.

se mide el tiempo con cudaEvent para tener la medida exacta de la gpu sin pasar por un procesamiento cpu


## conclusiones

El script el programa con distintos tamaños de N (hasta 16 millones) para asegurarme de saturar la GPU.

De los graficos se ve que el Modo 0 es muchísimo más rápido que el Modo 1. La diferencia se ve que crece exponencialmenten el teimpo del modo 1 (Esto ocurre por el Memory Coalescing)

    En el Modo 0, como los hilos del mismo Warp acceden a datos contiguos, la GPU puede traer todos esos datos en una sola "transacción" de memoria. Es muy eficiente.

    En el Modo 1, al ser aleatorio, se rompe la coalescencia. Para leer los datos que necesitan los 32 hilos de un Warp, la memoria tiene que hacer muchas peticiones pequeñas y separadas a distintos lugares de la RAM de video.

En conclusión, aunque la cantidad de sumas es la misma en ambos modos, el cuello de botella (el limitante) es el ancho de banda de la memoria. Se desperdicia mucha transferencia de datos cuando no hay orden

sobre el Speedup se compararon los 2 casos entre sí, sin considerar el costo lineal de hacer esto (por tiempo)
Utilizando la fórmula $S = T_{desordenado} / T_{ordenado}$, observé que el Speedup se estabiliza en un valor considerable (aprox. 10x - 15x en mi hardware).
* Esto indica que el costo de no ordenar los datos es altísimo: el algoritmo tarda 10 veces más solo por esperar datos de la memoria, aunque la carga computacional (sumas) sea idéntica.
