# Procesamiento-Digital-de-Im-genes-Operador-de-Convoluci-n-Sobel-en-CPU-Secuencial-vs-GPU-Paralelo-

Este proyecto implementa y compara dos soluciones a un mismo problema bajo paradigmas computacionales distintos para resolver el problema de detección de bordes en imágenes mediante el Operador Sobel.

## 1. Arquitectura y Diseño de la Solución
Este procesamiento de imágenes sigue un flujo de transformación de datos dividida en tres etapas principales.

1- Imagen Color (PPM P6)

2- Lectura de Cabecera

3- ETAPA 1: Conversión a Escala de Grises

4- ETAPA 2: Convolución con Máscaras Sobel Gx y Gy

5- ETAPA 3: Magnitud de Gradiente y Truncado

6- Imagen de Bordes (PPM P6)

### Descripción del Proceso de Transformación e Integridad de Archivos

Filtros de Verificación de Archivo Correcto: Para garantizar la robustez del sistema, se hacen los siguientes protocolos:
- Valida que el archivo de entrada cuente con la firma binaria P6.
- Se implementa un algoritmo de filtrado de comentarios que descarta cualquier línea que inicie con   el carácter # y limpia los espacios del encabezado.
- Se verifica que las dimensiones (ancho y alto) sean estrictamente mayores a cero.
- Mediante la función archivo.gcount(), el programa valida que los bytes leídos en disco coincidan    exactamente con el tamaño del búfer, previniendo fallos por archivos truncados o corruptos.

1. Paso 1: Conversión a Escala de Grises: El operador Sobel evalúa cambios en la intensidad lumínica y no de color. Por lo que es necesario aplicar una ecuación de luminancia estandarizada ITU-R BT.601 que pondera la sensibilidad:
Y = 0.299 * R + 0.587 * G + 0.114 * B

2. El resultado decimal (float) se convierte mediante un truncado explícito (static_cast<unsigned char>) a un entero de 8 bits sin signo.

3. Paso 2: Operador de Convolución Sobel: Se procesan los espacios circundantes mediante el desplazamiento de ventanas de 3 * 3 píxeles, multiplicando la vecindad por los coeficientes horizontal Gx y vertical Gy.

4. Magnitud del Gradiente y Truncado: Se unifican los vectores x, y calculando la longitud de la hipotenusa mediante el Teorema de Pitágoras, esto para obtener la magnitud real de iluminación:
Magnitud = sqrt{Gx^2 + Gy^2}

5. Los resultados matemáticos que exceden los límites cromáticos > 255 se topan estrictamente a 255 antes de guardarse en el vector de salida.

## 2. Análisis del Paradigma y Librerías

### Solución A: Paradigma Secuencial (CPU Single-Thread)
- Librerías Utilizadas: <iostream> (comunicación por consola), <fstream> (flujo binario de archivos), <vector> (arreglos contiguos en el Heap), <cmath> (cómputo de la raíz cuadrada) y <chrono> (medición de tiempo de alta resolución).

- Mecanismo: Basado en el uso lineal de la memoria a través de ciclos iterativos for anidados estrictamente secuenciales. Un único núcleo físico de la CPU asume la carga total de trabajo, procesando un píxel tras otro de manera secuencial.

### Solución B: Paradigma Paralelo y Concurrente (GPU - NVIDIA CUDA)
- Librerías y APIs de CUDA Utilizadas: Se incluye <cuda_runtime.h> para interactuar con la API de ejecución de NVIDIA. Se utilizan funciones de gestión de memoria de video (VRAM) como cudaMalloc (reserva de memoria global en el dispositivo), cudaMemcpy y cudaMemset para la inicialización de estructuras.

- Mecanismo: El problema se abstrae bajo el modelo usando multiples threads para atacar un mismo trabajo. El plano tridimensional (RGB) de la imagen se segmenta en una malla bidimensional de bloques compuestos por 16 * 16 hilos independientes 256 hilos por bloque. Se eliminan los bucles y la GPU lanza miles de hilos de forma simultánea, calculando las coordenadas mediante los registros del procesador:

<img width="1763" height="761" alt="image" src="https://github.com/user-attachments/assets/e4d95d0b-9020-418b-bec4-51962a3d31c4" />


FRAGMENTO DEL CÓDIGO:

int x = blockIdx.x * blockDim.x + threadIdx.x;

int y = blockIdx.y * blockDim.y + threadIdx.y;

- Los hilos ejecutan los calificadores __global__ denominados kernelEscalaGrises y kernelSobel. Se implementan eventos especializados (cudaEvent_t, cudaEventRecord, cudaEventElapsedTime) para capturar el tiempo de ejecución puro de los núcleos.

## 3. Análisis de Complejidad Computacional
Siendo W el ancho de la imagen y H el alto de la imagen. El número total de píxeles a procesar se define como N = W * H.

### Complejidad Temporal
- Secuencial (CPU): O(W * H) -> O(N). El algoritmo recorre cada píxel individualmente ejecutando una ventana de tamaño constante 3 * 3 = 9 operaciones. El tiempo de ejecución crece en proporción directa a la resolución total de la imagen.

- Paralelo (CUDA): O((W * H) / P) -> O(N / P) ideal. Al segmentar el procesamiento en hilos concurrentes, el tiempo de cómputo se divide por el número total de núcleos de ejecución paralela “ P “ en la GPU.

### Complejidad Espacial
- Ambos Paradigmas: O(W * H) -> O(N). Requieren instanciar arreglos de tamaño dinámico proporcional al volumen de píxeles de entrada para almacenar las estructuras intermedias y de salida en memoria:
-   Memoria Imagen Color: W * H * 3 bytes( sizeof(Pixel) ).
-   Memoria Imagen Gris / Bordes: W * H * 1 byte( sizeof(unsigned char) ).

## 4. Reporte de Pruebas y Rendimiento 
### Metodología de Pruebas
Las pruebas de rendimiento se ejecutaron utilizando tres imágenes de prueba con resoluciones distintas en formato PPM (P6). El tiempo medido contempla de forma estricta el procesamiento puro, aislando tiempos de lectura y escritura.

### Tabla Comparativa de Rendimiento
A continuación se detallan los tiempos promedio de ejecución obtenidos tras 3 iteraciones consecutivas por cada archivo de prueba:
| ID de Prueba | Archivo de Prueba | Resolución de Imagen | Tiempo CPU Secuencial | Tiempo GPU CUDA (Paralelo) |
| :--- | :--- | :---: | :---: | :---: |
| Prueba 01 | Hello_World!-input.jpg | 1170 x 585 | 111.156 ms | 0.182368 ms |
| Prueba 02 | cenicienta-input.jpg | 1024 x 640 | 61.5425 ms | 0.086010 ms |
| Prueba 03 | balloonerism-input.jpg | 3600 x 3600 | 832.23 ms | 1.31686 ms |


### Solución más óptima
El Paradigma Paralelo/Concurrente (NVIDIA CUDA) resalta en la comparativa de optimización. Mientras que el enfoque secuencial en CPU sufre en rendimiento por su paradigma lineal, escalando aún más al procesar imágenes de alta definición, la implementación en GPU absorbe masivamente la carga informática manteniendo los tiempos estables.
Esto demuestra que la aceleración del programa se maximiza al paralelizar, específicamente erradicando un bucle anidado que tiene un alto costo temporal , justificando la adopción de una arquitectura de hilos de GPU en el problema.

## 5. Visualización de Resultados Esperados
Aunque el output del procesamiento de las imágenes de las pruebas es brindado en formato ppm, dentro del repositorio se encuentran los resultados convertidos a formato jpg para una mejor visualización de los mismos, junto con su archivo en formato ppm P6, se encuentran como:


- Hello_World!-output.jpg
- cenicienta-output.jpg
- ballonerism-output.jpg

## 6. Manual de Configuración y Compilación en Google Colab
Para entornos que carecen localmente de una tarjeta gráfica NVIDIA física o un entorno local con el CUDA Toolkit, se detalla la configuración y el uso del compilador oficial de CUDA (nvcc) utilizando la infraestructura de cómputo en la nube de Google Colab.

Cabe aclarar que el programa secuencial en C++ puede correrse localmente, independientemente del hardware que se posea, ya que no requiere de múltiples cores para funcionar debidamente.

### Paso 1: Configurar el Entorno con Aceleración de Hardware GPU
1- Abrir un cuaderno nuevo en Google Colab.

2- Ir al menú superior: Entorno de ejecución -> Cambiar tipo de entorno de ejecución.

3- En la sección de "Acelerador de hardware", selecciona obligatoriamente T4 GPU (o cualquier GPU disponible). Da clic en Guardar.

### Paso 2: Verificar la Integridad del Compilador y Tarjeta Asignada
Ejecuta la siguiente celda de comandos para verificar que la suite de desarrollo de NVIDIA esté lista para su ejecución en la máquina virtual:

!nvidia-smi
!nvcc --version

### Paso 3: Escritura de Código y Compilación
Para compilar y correr las dos arquitecturas dentro de Colab, escribir el código precedido del comando %%writefile en la parte superior de la celda de la siguiente manera:
- Para compilar y ejecutar el Paradigma Secuencial:

%%writefile secuencial_sobel.cpp
// (Pegar aquí el código de C++)

EN OTRA CELDA:

!g++ -O3 secuencial_sobel.cpp -o filtro_secuencial
!./filtro_secuencial


- Para compilar y ejecutar el Paradigma Paralelo con CUDA:

%%writefile cuda_sobel.cu
// (Pegar aquí el código de CUDA)

EN OTRA CELDA:

!nvcc -arch=sm_75 cuda_sobel.cu -o filtro_cuda
!./filtro_cuda


## 7. Referencias Formales Bibliográficas
- Gonzales, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson. (Análisis formal matemático detrás del filtro espacial de gradiente de Sobel y la convolución discreta).
- Sanders, J., & Kandrot, E. (2010). CUDA by Example: An Introduction to General-Purpose GPU Programming. Addison-Wesley Professional. (Modelado y diseño del paradigma SIMT, asignación de bloques e hilos concurrentes).
- Sucar, L. E. y Gómez, G. (2011). Visión computacional. Instituto Nacional de Astrofísica, Óptica y Electrónica. 

