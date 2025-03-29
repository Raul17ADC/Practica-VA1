import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import json

def preprocesarImagen(imgPath):
    # Construir la ruta completa de la imagen
    completePath = os.path.join("Images", "testing", imgPath)

    img = cv2.imread(completePath, 0)  # Leer la imagen en escala de grises

    # Calcular el histograma de la imagen
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    # Suavizar el histograma usando un filtro gaussiano
    sigma = 2
    smoothed_hist = gaussian_filter1d(hist, sigma=sigma)

    # Encontrar los picos en el histograma suavizado
    peaks, _ = find_peaks(smoothed_hist, prominence=500)

    # Seleccionar el valor mínimo entre los dos máximos locales
    if len(peaks) >= 2:
        # Ordenar los picos por intensidad descendente
        peaks_sorted = sorted(peaks, key=lambda x: smoothed_hist[x], reverse=True)

        # Tomar los dos picos más altos
        p1, p2 = peaks_sorted[:2]

        if (abs(p1 - p2) < 30 and min(p1,p2) > 150):
          p1 = peaks_sorted[1] #Cogemos el segundo máximo
          p2 = peaks_sorted[2] #Cogemos el tercer máximo

        # Asegurar que p1 < p2 para buscar el mínimo entre ellos
        p1, p2 = min(p1, p2), max(p1, p2)

        # Encontrar el mínimo entre los dos picos más altos
        valley_index = np.argmin(smoothed_hist[p1:p2]) + p1
        threshold = valley_index
    else:
        threshold = 128  # Valor por defecto si no hay suficientes picos

    # Aplicar umbralización con el valor calculado
    _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Apertura con kernel 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    eroded_image = cv2.erode(thresholded_img, kernel)
    result_image = cv2.dilate(eroded_image, kernel)

    #Cierre con kernel de elipse 20x20
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    result_image = cv2.dilate(result_image, kernel)
    result_image = cv2.erode(result_image, kernel)

     # Apertura con kernel 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    eroded_image = cv2.erode(result_image, kernel)
    result_image = cv2.dilate(eroded_image, kernel)

    #Cierre con kernel de elipse 20x20
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
    result_image = cv2.dilate(result_image, kernel)
    result_image = cv2.erode(result_image, kernel)
    return result_image

def detectar_formas(imgPath):
    # Definir umbral mínimo de longitud de lado
    UMBRAL_MIN_LADO = 600  # Ajusta este valor según lo que consideres "pequeño"

    # Cargar imagen en escala de grises
    img = preprocesarImagen(imgPath)

    # Aplicar umbral si no está binaria
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear imágenes para visualizar resultados
    img_result = np.zeros_like(thresh)  # Imagen en negro para pintar el rectángulo corregido
    img_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # Imagen en color para visualizar los contornos

    # Buscar el contorno más grande
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Aproximar el contorno a una forma poligonal
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Dibujar el contorno detectado en la imagen original (verde)
        cv2.drawContours(img_color, [approx], 0, (0, 255, 0), 30)

        # Comprobamos si se detectaron al menos 4 puntos
        if len(approx) >= 4:
            # Obtener los cuatro puntos detectados
            puntos = approx[:, 0, :]  # Extraer coordenadas (x, y)

            # Ordenar puntos por posición Y (para diferenciar arriba y abajo)
            puntos = sorted(puntos, key=lambda p: (p[1], p[0]))
            arriba = sorted(puntos[:2], key=lambda p: p[0])  # Ordenar por X
            abajo = sorted(puntos[-2:], key=lambda p: p[0])  # Últimos 2 puntos para la parte inferior

            # Si hay más de 4 puntos, aplicar la selección de esquinas
            if len(approx) > 4:
                puntos_seleccionados = seleccionar_esquinas(puntos, img.shape)
            else:
                # Si hay 4 puntos, se usan tal cual están
                puntos_seleccionados = [arriba[0], arriba[1], abajo[1], abajo[0]]

            # Construir el nuevo contorno con las esquinas seleccionadas
            nuevo_contorno = np.array([puntos_seleccionados[0], puntos_seleccionados[1], puntos_seleccionados[2], puntos_seleccionados[3]], dtype=np.int32)

            # Dibujar el nuevo contorno en blanco en la imagen de resultado
            cv2.drawContours(img_result, [nuevo_contorno], 0, 255, thickness=cv2.FILLED)
        else:
            print("No se detectaron 4 vértices en el contorno más grande.")

    else:
        print("No se detectaron contornos en la imagen.")

    return img_result  # Devolver la imagen procesada

def seleccionar_esquinas(puntos, shape):
    # Obtener el centro de la imagen
    centro = (shape[1] // 2, shape[0] // 2)

    # Calcular las distancias al centro para cada punto
    distancias = [np.linalg.norm(np.array(p) - np.array(centro)) for p in puntos]

    # Ordenar los puntos por su distancia al centro (las esquinas más alejadas del centro serán las primeras)
    puntos_ordenados = [p for _, p in sorted(zip(distancias, puntos), reverse=True)]

    # Seleccionar las 4 esquinas más alejadas del centro
    return puntos_ordenados[:4]

def getListaImgEsquinas(jsonFileName):
    # Construir la ruta completa del archivo JSON
    jsonPath = os.path.join("data", jsonFileName)
    #print("Leyendo JSON desde:", jsonPath)

    # Cargar el archivo JSON
    with open(jsonPath, "r") as file:
        data = json.load(file)

    # Obtener los datos de "_via_img_metadata"
    img_metadata = data["_via_img_metadata"]

    # Lista para almacenar las imágenes con sus puntos
    listaImagenPuntos = []

    # Recorrer cada imagen en los metadatos
    for img_id, img_info in img_metadata.items():
        filename = img_info["filename"]  # Obtener el nombre de la imagen

        # Obtener las coordenadas de las esquinas
        puntos = [(region["shape_attributes"]["cx"], region["shape_attributes"]["cy"]) for region in img_info["regions"]]

        # Agregar la tupla (nombre_imagen, lista_puntos) a la lista final
        listaImagenPuntos.append((filename, puntos))

    return listaImagenPuntos

def getListaDescriptoresLearning(listaImagenPuntos):
    train_descriptors = []
    # Inicializar SIFT
    sift = cv2.SIFT_create()
    for i in range(len(listaImagenPuntos)):
        ruta = listaImagenPuntos[i][0]
        esquinas = listaImagenPuntos[i][1]

        # Construir la ruta completa de la imagen
        completePath = os.path.join("Images", "learning", ruta)

        img = cv2.imread(completePath,0)
        keypoints = [cv2.KeyPoint(float(x), float(y), 10) for x, y in esquinas]  # Convertir coordenadas en KeyPoint
        # Detectar keypoints y computar descriptores
        keypoints, descriptors = sift.compute(img, keypoints)
        train_descriptors.append(descriptors)

    train_descriptors = np.vstack(train_descriptors)
    return train_descriptors

#preprocesarImagen("Hoja_01.jpg")
#listaImgEsquinas = getListaImgEsquinas("esquinas_learning.json")
#print(listaImgEsquinas)
#listaDescriptoresEntrenamiento = getListaDescriptoresLearning(listaImgEsquinas)
#print("Lista de descriptores calculados")

def mostrarEsquinasDetectadas(ruta, esquinas):
    """Muestra los puntos detectados como esquinas en la imagen."""
    # Construir la ruta completa de la imagen
    completePath = os.path.join("Images", "testing", ruta)
    img_color = cv2.imread(completePath)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_kp = img_color.copy()
    #print("Nombre de la imagen: ",completePath)

    for x, y in esquinas:
        x, y = int(x), int(y)  # Convertir a enteros
        cv2.circle(img_kp, (x, y), 60, (0, 255, 0), -1)  # Dibujar círculo verde

    plt.imshow(img_kp)
    plt.axis("off")
    plt.show()

def dividirImagen(ruta, descriptoresEntrenamiento):
    """Divide la imagen en cuadrantes, encuentra los keypoints y selecciona la mejor esquina por cuadrante."""
    # Construir la ruta completa de la imagen
    completePath = os.path.join("Images", "testing", ruta)
    # Cargar la imagen
    #img_color = cv2.imread(ruta)
    #img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    #img = cv2.imread(ruta, 0)  # Imagen en escala de grises

    img = detectar_formas(ruta)
    #plt.imshow(img, cmap="gray")
    #plt.axis("off")
    #plt.show()
    # Detectar keypoints con FAST
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)
    kp = fast.detect(img, None)

    # Obtener dimensiones
    height, width = img.shape

    # Dividir keypoints en cuadrantes
    cuadrantes = {
        "PC": [],  # Primer cuadrante (arriba izquierda)
        "SC": [],  # Segundo cuadrante (arriba derecha)
        "TC": [],  # Tercer cuadrante (abajo izquierda)
        "CC": []   # Cuarto cuadrante (abajo derecha)
    }

    for keypoint in kp:
        x, y = keypoint.pt
        if y < height // 2 and x < width // 2:
            cuadrantes["PC"].append(keypoint)
        elif y < height // 2 and x > width // 2:
            cuadrantes["SC"].append(keypoint)
        elif y > height // 2 and x < width // 2:
            cuadrantes["TC"].append(keypoint)
        elif y > height // 2 and x > width // 2:
            cuadrantes["CC"].append(keypoint)

    # Mostrar los keypoints de cada cuadrante
    #for cuadrante in cuadrantes:
      #mostrarEsquinasDetectadas(ruta, [kp.pt for kp in cuadrantes[cuadrante]])

    # Extraer descriptores por cuadrante
    # Inicializar SIFT
    sift = cv2.SIFT_create()
    # Cambiamos BFMatcher para usar la distancia euclídea
    bf = cv2.BFMatcher(cv2.NORM_L2)

    mejores_puntos = []  # Guardaremos las 4 mejores esquinas aquí

    for nombre, keypoints in cuadrantes.items():
        if len(keypoints) == 0:
            continue  # Si no hay keypoints en este cuadrante, lo saltamos

        # Extraer descriptores
        kp, des = sift.compute(img, keypoints)

        if des is None:
            continue  # Si no se encuentran descriptores, saltamos este cuadrante

        # Realizar KNN Match con los descriptores de entrenamiento
        matches = bf.knnMatch(des, descriptoresEntrenamiento, k=2)

        # Encontrar el mejor match en el cuadrante (menor distancia)
        menor_distancia = float("inf")
        mejor_keypoint = None

        for m, n in matches:
            if m.distance < 0.95 * n.distance and m.distance < menor_distancia:
                menor_distancia = m.distance
                mejor_keypoint = kp[m.queryIdx]  # Guardamos el keypoint con mejor match

        if mejor_keypoint:
            mejores_puntos.append(mejor_keypoint.pt)  # Guardamos coordenadas (x, y)

    if(len(mejores_puntos) > 3):
        rectificar_imagen(ruta, mejores_puntos)
    else:
        print(f"Advertencia: No se han encontrado suficientes esquinas en la imagen '{ruta}'.")

def rectificar_imagen(ruta, mejores_puntos):
    completePath = os.path.join("Images", "testing", ruta)
    img_color = cv2.imread(completePath)

    puntos = np.array(mejores_puntos, np.float32)
    puntos[[2, 3]] = puntos[[3, 2]]  # Intercambia el tercer y cuarto punto

    if(len(mejores_puntos) > 3):
        '''
        ancho_sup = np.sqrt((puntos[1][0] - puntos[0][0])**2 + (puntos[1][1] - puntos[0][1])**2) #distancia ancho superior
        ancho_inf = np.sqrt((puntos[2][0] - puntos[3][0])**2 + (puntos[2][1] - puntos[3][1])**2) #distancia ancho inferior

        ancho_max = max(ancho_sup, ancho_inf) #ancho mayor
        '''
        ancho_sup = np.linalg.norm(puntos[1] - puntos[0])
        ancho_inf = np.linalg.norm(puntos[2] - puntos[3])

        ancho_max = max(int(ancho_sup), int(ancho_inf))

        nuevo_alto = int(1.41 * ancho_max)

        #puntos finales
        p_destino = np.array([[0, 0],[ancho_max, 0],[ancho_max, nuevo_alto],[0, nuevo_alto]], np.float32)

        #calculamos la matriz de redimensionado
        matriz = cv2.getPerspectiveTransform(puntos, p_destino)
        imagen_final = cv2.warpPerspective(img_color, matriz,(ancho_max, nuevo_alto))
        plt.imshow(cv2.cvtColor(imagen_final, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

# Llamada a la función con imágenes de prueba
"""dividirImagen("Hoja_01.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_02.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_03.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_04.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_05.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_06.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_07.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_08.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_09.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_10.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_11.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_12.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_13.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_14.jpg", listaDescriptoresEntrenamiento)
dividirImagen("Hoja_15.jpg", listaDescriptoresEntrenamiento)"""

def main():
    if len(sys.argv) != 2:
        print("Uso: python <nombre_imagen_sin_extension>")
        sys.exit(1)

    imagen_path = sys.argv[1]  # Tomamos el argumento sin extensión

    # Verifica que la imagen exista en Images/testing/
    ruta_completa = os.path.join("Images", "testing", imagen_path)
    if not os.path.exists(ruta_completa):
        print(f"Error: No se encontró el archivo {ruta_completa}")
        sys.exit(1)

    # Cargar los descriptores previamente calculados
    listaImgEsquinas = getListaImgEsquinas("esquinas_learning.json")
    listaDescriptoresEntrenamiento = getListaDescriptoresLearning(listaImgEsquinas)

    # Llamar a dividirImagen con la imagen proporcionada
    dividirImagen(imagen_path, listaDescriptoresEntrenamiento)

if __name__ == "__main__":
    main()