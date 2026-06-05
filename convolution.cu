/**
 * @file convolution.cu
 * @brief Filtro de detección de bordes Sobel utilizando el Paradigma Paralelo/Concurrente.
 * @details Este código transfiere la matriz de píxeles a la VRAM de la GPU y ejecuta
 * kernels paralelos masivos donde cada hilo mapea un píxel tridimensional independiente.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Estructura alineada para representación de color en Host
struct Pixel {
    unsigned char r, g, b;
};

/**
 * @brief Kernel de CUDA encargado de paralelizar la conversión a escala de grises.
 * @param d_color Puntero a la memoria global de la GPU con los datos RGB.
 * @param d_gris Puntero donde se almacenará el resultado monocromático.
 */
__global__ void kernelEscalaGrises(const Pixel* d_color, unsigned char* d_gris, int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < ancho && y < alto) {
        int idx = y * ancho + x;
        // Operación matemática en paralelo sin interferencia de memoria de otros hilos
        d_gris[idx] = (unsigned char)(0.299f * d_color[idx].r + 
                                      0.587f * d_color[idx].g + 
                                      0.114f * d_color[idx].b);
    }
}

/**
 * @brief Kernel de CUDA que paraleliza el operador convolucional de Sobel.
 */
__global__ void kernelSobel(const unsigned char* d_gris, unsigned char* d_bordes, int ancho, int alto) {
    // Linearización de punteros
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Evitar procesar los bordes extremos del marco para prevenir accesos fuera de límites
    if (x > 0 && x < ancho - 1 && y > 0 && y < alto - 1) {
        
        // Almacenamiento local por thread para las matrices convolucionales
        float sumaX = 0.0f;
        float sumaY = 0.0f;

        // Máscaras de Sobel cargadas en registros locales del hilo
        // bordes verticales
        int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        // bordes horizontales
        int Gy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        // Convolución local de vecindad de forma concurrente con el resto de la malla
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                unsigned char val = d_gris[(y + ky) * ancho + (x + kx)];
                sumaX += val * Gx[ky + 1][kx + 1];
                sumaY += val * Gy[ky + 1][kx + 1];
            }
        }

        // truncar a 255 por bordes fuertes (cambios drásticos)
        float magnitud = sqrtf(sumaX * sumaX + sumaY * sumaY);
        d_bordes[y * ancho + x] = (magnitud > 255.0f) ? 255 : (unsigned char)magnitud;
    }
}

// Funciones auxiliares de lectura/escritura en Host (CPU)
bool leerPPM(const std::string& ruta, int& ancho, int& alto, std::vector<Pixel>& imagen) {

    //verificar archivo pueda abrirse y tenga formato correcto
    std::ifstream archivo(ruta, std::ios::binary);
    if (!archivo.is_open()) return false;
    std::string formato;
    archivo >> formato;
    if (formato != "P6") return false;
    
    // Saltar comentarios y espacios en blanco
    char c = archivo.peek();
    while (c == '\n' || c == '\r' || c == ' ') { archivo.get(); c = archivo.peek(); }
    if (c == '#') { std::string com; std::getline(archivo, com); }
    
    // Leer dimensiones y valor máximo de color
    int maxVal;
    archivo >> ancho >> alto >> maxVal;
    archivo.get();

    // Leer datos de píxeles en formato binario
    imagen.resize(ancho * alto);
    archivo.read(reinterpret_cast<char*>(imagen.data()), imagen.size() * sizeof(Pixel));
    return archivo.good();
}

bool guardarPPM(const std::string& ruta, int ancho, int alto, const std::vector<unsigned char>& grises) {
    // Escribir encabezado y datos de píxeles en formato binario
    std::ofstream archivo(ruta, std::ios::binary);
    if (!archivo.is_open()) return false;
    archivo << "P6\n" << ancho << " " << alto << "\n255\n";
    std::vector<Pixel> salida(ancho * alto);
    for (int i = 0; i < ancho * alto; ++i) {
        salida[i].r = grises[i]; salida[i].g = grises[i]; salida[i].b = grises[i];
    }
    archivo.write(reinterpret_cast<const char*>(salida.data()), salida.size() * sizeof(Pixel));
    return archivo.good();
}

int main() {
    int ancho = 0, alto = 0;
    std::vector<Pixel> h_imagenColor;

    if (!leerPPM("input.ppm", ancho, alto, h_imagenColor)) {
        std::cerr << "Error al abrir input.ppm" << std::endl;
        return 1;
    }

    size_t tamanoColor = ancho * alto * sizeof(Pixel);
    size_t tamanoGris = ancho * alto * sizeof(unsigned char);

    // Vectores en Host para almacenar el resultado final devuelto por la GPU
    std::vector<unsigned char> h_bordes(ancho * alto, 0);

    // Punteros de Memoria del Dispositivo
    Pixel* d_color = nullptr;
    unsigned char* d_gris = nullptr;
    unsigned char* d_bordes = nullptr;

    // Asignación de memoria dinámica en GPU
    cudaMalloc((void**)&d_color, tamanoColor);
    cudaMalloc((void**)&d_gris, tamanoGris);
    cudaMalloc((void**)&d_bordes, tamanoGris);

    // Eventos de CUDA para medir el tiempo exacto de procesamiento en hardware
    cudaEvent_t inicio, fin;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fin);

    // Copiar datos de entrada desde CPU (Host) hacia la GPU (Device)
    cudaMemcpy(d_color, h_imagenColor.data(), tamanoColor, cudaMemcpyHostToDevice);
    // Inicializar memoria de bordes en 0
    cudaMemset(d_bordes, 0, tamanoGris);

    // Configuración de la Malla de Hilos
    // Bloques bidimensionales de 16x16 hilos (256 hilos por bloque)
    dim3 dimensionesBloque(16, 16);
    dim3 dimensionesGrid((ancho + dimensionesBloque.x - 1) / dimensionesBloque.x,
                         (alto + dimensionesBloque.y - 1) / dimensionesBloque.y);

    // Iniciar grabación de tiempo del Dispositivo
    cudaEventRecord(inicio);

    // Lanzamiento concurrente del Kernel 1: Conversión a Gris
    kernelEscalaGrises<<<dimensionesGrid, dimensionesBloque>>>(d_color, d_gris, ancho, alto);
    
    // Lanzamiento concurrente del Kernel 2: Filtro Sobel
    kernelSobel<<<dimensionesGrid, dimensionesBloque>>>(d_gris, d_bordes, ancho, alto);

    // Detener grabación de tiempo y sincronizar hilos
    cudaEventRecord(fin);
    cudaEventSynchronize(fin);

    float tiempoCuda = 0;
    cudaEventElapsedTime(&tiempoCuda, inicio, fin);
    std::cout << "Procesamiento en GPU completado en: " << tiempoCuda << " ms" << std::endl;

    // Transferir el resultado final de vuelta desde la GPU hacia la memoria RAM (Host)
    cudaMemcpy(h_bordes.data(), d_bordes, tamanoGris, cudaMemcpyDeviceToHost);

    // Liberación de memoria
    cudaFree(d_color);
    cudaFree(d_gris);
    cudaFree(d_bordes);
    cudaEventDestroy(inicio);
    cudaEventDestroy(fin);

    if (guardarPPM("output_cuda.ppm", ancho, alto, h_bordes)) {
        std::cout << "Imagen guardada como output_cuda.ppm" << std::endl;
    }

    return 0;
}