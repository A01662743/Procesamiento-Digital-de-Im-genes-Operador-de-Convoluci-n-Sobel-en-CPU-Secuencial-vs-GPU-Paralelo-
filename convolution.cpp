/**
 * @file secuencial_sobel.cpp
 * @brief Filtro de detección de bordes Sobel utilizando el Paradigma Secuencial.
 * @details El programa procesa una imagen a color en formato PPM línea por línea,
 * realiza una conversión a escala de grises y aplica convolución de matrices 3x3
 * de forma estrictamente lineal (Single-Thread) en CPU.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

// Estructura para representar un píxel a color (RGB de 8 bits por canal)
struct Pixel {
    unsigned char r, g, b;
};

/**
 * @brief Lee una imagen en formato PPM (P6) desde disco.
 */
bool leerPPM(const std::string& ruta, int& ancho, int& alto, std::vector<Pixel>& imagen) {
    std::ifstream archivo(ruta, std::ios::binary);
    if (!archivo.is_open()) {
        std::cout << "[ERROR] No se pudo abrir: " << ruta << std::endl;
        return false;
    }

    std::string formato;
    archivo >> formato;
    if (formato != "P6") {
        std::cout << "[ERROR] Formato no es P6: " << formato << std::endl;
        return false;
    }

    // Saltar todos los comentarios y espacios correctamente
    char c;
    while (archivo.get(c)) {
        if (c == '#') {
            // Ignorar línea completa de comentario
            while (archivo.get(c) && c != '\n');
        } else if (!isspace((unsigned char)c)) {
            // Primer carácter real del encabezado: devolverlo
            archivo.putback(c);
            break;
        }
    }

    if (!(archivo >> ancho >> alto)) {
        std::cout << "[ERROR] No se pudo leer ancho/alto." << std::endl;
        return false;
    }

    int maxValor;
    if (!(archivo >> maxValor)) {
        std::cout << "[ERROR] No se pudo leer valor máximo." << std::endl;
        return false;
    }

    // Consumir TODOS los espacios/saltos tras el "255"
    while (archivo.peek() != EOF && isspace((unsigned char)archivo.peek())) {
        archivo.get();
    }

    if (ancho <= 0 || alto <= 0) {
        std::cout << "[ERROR] Dimensiones inválidas: " << ancho << "x" << alto << std::endl;
        return false;
    }

    // resize vector para almacenar los píxeles y leer datos binarios
    imagen.resize(ancho * alto);
    archivo.read(reinterpret_cast<char*>(imagen.data()), imagen.size() * sizeof(Pixel));

    // Verificar que se hayan leído suficientes datos de píxeles
    if (archivo.gcount() < (std::streamsize)(imagen.size() * sizeof(Pixel))) {
        std::cout << "[ERROR] Archivo truncado. Leídos: " << archivo.gcount() 
                  << " de " << imagen.size() * sizeof(Pixel) << " bytes." << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Escribe una imagen en formato PPM (P6) a disco (escala de grises guardada como RGB).
 */
bool guardarPPM(const std::string& ruta, int ancho, int alto, const std::vector<unsigned char>& grises) {
    std::ofstream archivo(ruta, std::ios::binary);
    if (!archivo.is_open()) return false;

    archivo << "P6\n" << ancho << " " << alto << "\n255\n";

    // Convertir el canal único de gris de vuelta a un formato RGB para que sea un PPM válido
    std::vector<Pixel> salida(ancho * alto);
    for (int i = 0; i < ancho * alto; ++i) {
        salida[i].r = grises[i];
        salida[i].g = grises[i];
        salida[i].b = grises[i];
    }

    archivo.write(reinterpret_cast<const char*>(salida.data()), salida.size() * sizeof(Pixel));
    return archivo.good();
}

int main() {
    int ancho = 0, alto = 0;
    std::vector<Pixel> imagenColor;

    std::cout << "[Secuencial] Leyendo imagen input.ppm..." << std::endl;
    if (!leerPPM("input.ppm", ancho, alto, imagenColor)) {
        std::cout << "Error: No se pudo leer la imagen input.ppm o no es un formato P6 válido." << std::flush << std::endl;
        return 1;
    }

    // Iniciar medición de tiempo del procesamiento algorítmico
    auto inicio = std::chrono::high_resolution_clock::now();

    //Conversión a Escala de Grises
    std::vector<unsigned char> imagenGris(ancho * alto);
    for (int i = 0; i < ancho * alto; ++i) {
        // Fórmula de luminancia estándar ITU-R BT.601
        imagenGris[i] = static_cast<unsigned char>(0.299f * imagenColor[i].r + 
                                                   0.587f * imagenColor[i].g + 
                                                   0.114f * imagenColor[i].b);
    }

    // Operador Sobel
    std::vector<unsigned char> bordes(ancho * alto, 0);

    // Kernels de Sobel para gradientes horizontales (Gx) y verticales (Gy)
    const int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Procesar píxeles internos (evitando bordes externos)
    for (int y = 1; y < alto - 1; ++y) {
        for (int x = 1; x < ancho - 1; ++x) {
            float sumaX = 0.0f;
            float sumaY = 0.0f;

            // Convolución de la vecindad 3x3
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixelVecino = imagenGris[(y + ky) * ancho + (x + kx)];
                    sumaX += pixelVecino * Gx[ky + 1][kx + 1];
                    sumaY += pixelVecino * Gy[ky + 1][kx + 1];
                }
            }

            // Calcular magnitud del gradiente (calcular el brillo del borde y en positivo, independiente a la dirección)
            float magnitud = std::sqrt(sumaX * sumaX + sumaY * sumaY);
            
            // Truncar valor al rango de 8 bits por bordes marcados (cambios drásticos)
            bordes[y * ancho + x] = (magnitud > 255.0f) ? 255 : static_cast<unsigned char>(magnitud);
        }
    }

    auto fin = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tiempoS = fin - inicio;

    std::cout << "[Secuencial] Procesamiento completado en: " << tiempoS.count() << " ms" << std::endl;

    if (guardarPPM("output_secuencial.ppm", ancho, alto, bordes)) {
        std::cout << "[Secuencial] Imagen guardada exitosamente como output_secuencial.ppm" << std::endl;
    } else {
        std::cerr << "Error al guardar la imagen de salida." << std::endl;
    }

    return 0;
}