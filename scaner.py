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
    def preprocess(image_name):
        path = os.path.join("Images", "testing", image_name)
        img = cv2.imread(path, 0)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        smoothed = gaussian_filter1d(hist, sigma=2)
        peaks, _ = find_peaks(smoothed, prominence=500)

        if len(peaks) >= 2:
            peaks = sorted(peaks, key=lambda x: smoothed[x], reverse=True)[:3]
            p1, p2 = sorted(peaks[:2])
            valley = np.argmin(smoothed[p1:p2]) + p1
            thresh = valley
        else:
            thresh = 128

        _, bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
        k4 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))

        img = cv2.erode(bin_img, k1)
        img = cv2.dilate(img, k1)
        img = cv2.dilate(img, k2)
        img = cv2.erode(img, k2)
        img = cv2.erode(img, k3)
        img = cv2.dilate(img, k3)
        img = cv2.dilate(img, k4)
        img = cv2.erode(img, k4)

        return img


class DescriptorLoader:
    @staticmethod
    def load_from_json(filename):
        esquinas = []
        with open(os.path.join("data", filename)) as f:
            data = json.load(f)["_via_img_metadata"]
            for img in data.values():
                puntos = [
                    (r["shape_attributes"]["cx"], r["shape_attributes"]["cy"])
                    for r in img["regions"]
                ]
                esquinas.append((img["filename"], puntos))
        return DescriptorLoader.compute_descriptors(esquinas)

    @staticmethod
    def compute_descriptors(listaImagenPuntos):
        sift = cv2.SIFT_create()
        descriptores = []
        for nombre, puntos in listaImagenPuntos:
            ruta = os.path.join("Images", "learning", nombre)
            img = cv2.imread(ruta, 0)
            keypoints = [cv2.KeyPoint(x, y, 10) for x, y in puntos]
            _, desc = sift.compute(img, keypoints)
            descriptores.append(desc)
        return np.vstack(descriptores)


class CornerDetector:
    @staticmethod
    def detect(image_name, descriptors_ref):
        img_bin = ImagePreprocessor.preprocess(f"{image_name}.jpg")
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(False)
        keypoints = fast.detect(img_bin, None)

        cuadrantes = CornerDetector.divide_in_quadrants(keypoints, img_bin.shape)
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2)
        puntos_finales = []

        for kps in cuadrantes.values():
            if not kps:
                continue
            _, des = sift.compute(img_bin, kps)
            if des is None:
                continue
            matches = bf.knnMatch(des, descriptors_ref, k=2)
            mejor = min(
                (m for m, n in matches if m.distance < 0.95 * n.distance),
                default=None,
                key=lambda m: m.distance,
            )
            if mejor:
                puntos_finales.append(kps[mejor.queryIdx].pt)

        return puntos_finales

    @staticmethod
    def divide_in_quadrants(keypoints, shape):
        h, w = shape
        zonas = {"PC": [], "SC": [], "TC": [], "CC": []}
        for kp in keypoints:
            x, y = kp.pt
            if y < h // 2 and x < w // 2:
                zonas["PC"].append(kp)
            elif y < h // 2 and x > w // 2:
                zonas["SC"].append(kp)
            elif y > h // 2 and x < w // 2:
                zonas["TC"].append(kp)
            elif y > h // 2 and x > w // 2:
                zonas["CC"].append(kp)
        return zonas


class Rectifier:
    @staticmethod
    def rectify(image_path, puntos):
        img = cv2.imread(image_path)
        puntos = np.array(puntos, dtype=np.float32)
        puntos[[2, 3]] = puntos[[3, 2]]

        ancho_sup = np.linalg.norm(puntos[1] - puntos[0])
        ancho_inf = np.linalg.norm(puntos[2] - puntos[3])
        ancho = int(max(ancho_sup, ancho_inf))
        alto = int(1.41 * ancho)

        destino = np.array(
            [[0, 0], [ancho, 0], [ancho, alto], [0, alto]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(puntos, destino)
        result = cv2.warpPerspective(img, M, (ancho, alto))

        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()


class DocumentScanner:
    def __init__(self, image_name):
        self.image_name = image_name
        self.descriptors = None
        self.image_path = os.path.join("Images", "testing", f"{image_name}.jpg")

    def run(self):
        self.descriptors = DescriptorLoader.load_from_json("esquinas_learning.json")
        puntos = CornerDetector.detect(self.image_name, self.descriptors)
        if len(puntos) == 4:
            Rectifier.rectify(self.image_path, puntos)
        else:
            print("No se encontraron suficientes esquinas.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python scaner.py <nombre_imagen_sin_extension>")
        sys.exit(1)

    image_name = sys.argv[1]
    scanner = DocumentScanner(image_name)
    scanner.run()
