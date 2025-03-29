import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import json


class ImagePreprocessor:
    @staticmethod
    def preprocess(imgPath):
        completePath = os.path.join("Images", "testing", imgPath)
        img = cv2.imread(completePath, 0)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        smoothed_hist = gaussian_filter1d(hist, sigma=2)
        peaks, _ = find_peaks(smoothed_hist, prominence=500)

        if len(peaks) >= 2:
            peaks_sorted = sorted(peaks, key=lambda x: smoothed_hist[x], reverse=True)
            p1, p2 = peaks_sorted[:2]
            if abs(p1 - p2) < 30 and min(p1, p2) > 150:
                p1 = peaks_sorted[1]
                p2 = peaks_sorted[2]
            p1, p2 = min(p1, p2), max(p1, p2)
            valley_index = np.argmin(smoothed_hist[p1:p2]) + p1
            threshold = valley_index
        else:
            threshold = 128

        _, thresholded_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
        k4 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))

        img = cv2.erode(thresholded_img, k1)
        img = cv2.dilate(img, k1)
        img = cv2.dilate(img, k2)
        img = cv2.erode(img, k2)
        img = cv2.erode(img, k3)
        img = cv2.dilate(img, k3)
        img = cv2.dilate(img, k4)
        img = cv2.erode(img, k4)

        return img

class ShapeDetector:
    @staticmethod
    def detectar_formas(imgPath):
        UMBRAL_MIN_LADO = 600  # Ajusta según necesidad

        # Procesar la imagen (usando el preprocesamiento original)
        img = ImagePreprocessor.preprocess(imgPath)

        # Aplicar umbral si no está binaria
        _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Imagen para visualizar resultado
        img_result = np.zeros_like(thresh)
        img_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Aproximar el contorno
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            cv2.drawContours(img_color, [approx], 0, (0, 255, 0), 30)

            if len(approx) >= 4:
                puntos = approx[:, 0, :]
                puntos = sorted(puntos, key=lambda p: (p[1], p[0]))
                arriba = sorted(puntos[:2], key=lambda p: p[0])
                abajo = sorted(puntos[-2:], key=lambda p: p[0])

                if len(approx) > 4:
                    puntos_seleccionados = ShapeDetector.seleccionar_esquinas(puntos, img.shape)
                else:
                    puntos_seleccionados = [arriba[0], arriba[1], abajo[1], abajo[0]]

                nuevo_contorno = np.array(puntos_seleccionados, dtype=np.int32)
                cv2.drawContours(img_result, [nuevo_contorno], 0, 255, thickness=cv2.FILLED)
            else:
                print("No se detectaron 4 vértices en el contorno más grande.")
        else:
            print("No se detectaron contornos en la imagen.")

        return img_result

    @staticmethod
    def seleccionar_esquinas(puntos, shape):
        centro = (shape[1] // 2, shape[0] // 2)
        distancias = [np.linalg.norm(np.array(p) - np.array(centro)) for p in puntos]
        puntos_ordenados = [p for _, p in sorted(zip(distancias, puntos), reverse=True)]
        return puntos_ordenados[:4]



class DescriptorLoader:
    @staticmethod
    def load_from_json(jsonFileName):
        jsonPath = os.path.join("data", jsonFileName)
        with open(jsonPath, "r") as file:
            data = json.load(file)
        img_metadata = data["_via_img_metadata"]
        listaImagenPuntos = []
        for img_id, img_info in img_metadata.items():
            filename = img_info["filename"]
            puntos = [
                (region["shape_attributes"]["cx"], region["shape_attributes"]["cy"])
                for region in img_info["regions"]
            ]
            listaImagenPuntos.append((filename, puntos))
        return DescriptorLoader.compute_descriptors(listaImagenPuntos)

    @staticmethod
    def compute_descriptors(listaImagenPuntos):
        train_descriptors = []
        sift = cv2.SIFT_create()
        for nombre, puntos in listaImagenPuntos:
            completePath = os.path.join("Images", "learning", nombre)
            img = cv2.imread(completePath, 0)
            keypoints = [cv2.KeyPoint(float(x), float(y), 10) for x, y in puntos]
            keypoints, descriptors = sift.compute(img, keypoints)
            train_descriptors.append(descriptors)
        return np.vstack(train_descriptors)


class CornerDetector:
    @staticmethod
    def detect(ruta, descriptoresEntrenamiento):
        img = ShapeDetector.detectar_formas(ruta)
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(False)
        kp = fast.detect(img, None)

        height, width = img.shape
        cuadrantes = {"PC": [], "SC": [], "TC": [], "CC": []}

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

        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2)
        mejores_puntos = []

        for keypoints in cuadrantes.values():
            if len(keypoints) == 0:
                continue
            kp, des = sift.compute(img, keypoints)
            if des is None:
                continue
            matches = bf.knnMatch(des, descriptoresEntrenamiento, k=2)
            menor_distancia = float("inf")
            mejor_keypoint = None
            for m, n in matches:
                if m.distance < 0.95 * n.distance and m.distance < menor_distancia:
                    menor_distancia = m.distance
                    mejor_keypoint = kp[m.queryIdx]
            if mejor_keypoint:
                mejores_puntos.append(mejor_keypoint.pt)

        return mejores_puntos


class Rectifier:
    @staticmethod
    def rectify(ruta, mejores_puntos):
        completePath = os.path.join("Images", "testing", ruta)
        img_color = cv2.imread(completePath)
        puntos = np.array(mejores_puntos, np.float32)
        puntos[[2, 3]] = puntos[[3, 2]]

        if len(mejores_puntos) > 3:
            ancho_sup = np.linalg.norm(puntos[1] - puntos[0])
            ancho_inf = np.linalg.norm(puntos[2] - puntos[3])
            ancho_max = max(int(ancho_sup), int(ancho_inf))
            nuevo_alto = int(1.41 * ancho_max)
            p_destino = np.array(
                [[0, 0], [ancho_max, 0], [ancho_max, nuevo_alto], [0, nuevo_alto]],
                np.float32,
            )
            matriz = cv2.getPerspectiveTransform(puntos, p_destino)
            imagen_final = cv2.warpPerspective(
                img_color, matriz, (ancho_max, nuevo_alto)
            )
            plt.imshow(cv2.cvtColor(imagen_final, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()


class DocumentScanner:
    def __init__(self, descriptores):
        self.descriptores = descriptores

    def run(self, imagen_nombre):
        imagen_path = sys.argv[1]  # Tomamos el argumento sin extensión
        ruta_completa = os.path.join("Images", "testing", imagen_path)
        if not os.path.exists(ruta_completa):
            print(f"Error: No se encontró el archivo {ruta_completa}")
            return

        puntos = CornerDetector.detect(imagen_path, self.descriptores)
        if len(puntos) > 3:
            Rectifier.rectify(imagen_path, puntos)
        else:
            print(
                f"Advertencia: No se han encontrado suficientes esquinas en la imagen '{imagen_path}'."
            )


if __name__ == "__main__":
    descriptores = DescriptorLoader.load_from_json("esquinas_learning.json")
    scanner = DocumentScanner(descriptores)

    if len(sys.argv) == 2:
        # Procesamiento individual (por argumento)
        nombre_imagen = sys.argv[1]
        scanner.run(nombre_imagen)

    else:
        # Procesamiento por lotes (comentarlo si solo quieres procesamiento individual)
        print("Procesando todas las imágenes en Images/testing...")
        carpeta = os.path.join("Images", "testing")
        imagenes = [f for f in os.listdir(carpeta) if f.endswith(".jpg")]
        for img in sorted(imagenes):
            nombre_base = os.path.splitext(img)[0]
            scanner.run(nombre_base)
