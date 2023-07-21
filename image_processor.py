import numpy as np
import cv2
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def save_processed_image(self, image_name, processed_image):
        path = os.path.join(os.getcwd(), 'cv')
        if not os.path.exists(path):
            os.mkdir(path)
        image_path = os.path.join(path, image_name)
        cv2.imwrite(image_path, processed_image)

    def apply_median_filter(self, ksize=5):
        blurM = cv2.medianBlur(self.gray, ksize)
        self.save_processed_image('blurM.png', blurM)

    def apply_gaussian_filter(self, ksize=(9, 9)):
        blurG = cv2.GaussianBlur(self.gray, ksize, 0)
        self.save_processed_image('blurG.png', blurG)

    def histogram_equalization(self):
        histoNorm = cv2.equalizeHist(self.gray)
        self.save_processed_image('histoNorm.png', histoNorm)

    def clahe(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        claheNorm = clahe.apply(self.gray)
        self.save_processed_image('claheNorm.png', claheNorm)

    @staticmethod
    def pixel_val(pix, r1, s1, r2, s2):
        if 0 <= pix <= r1:
            return (s1 / r1) * pix
        elif r1 < pix <= r2:
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

    def contrast_stretching(self, r1, s1, r2, s2):
        pixel_val_vec = np.vectorize(self.pixel_val)
        contrast_stretched = pixel_val_vec(self.gray, r1, s1, r2, s2)
        self.save_processed_image('contrast_stretch.png', contrast_stretched)

    def canny_edge_detection(self, min_val, max_val):
        edge = cv2.Canny(self.gray, min_val, max_val)
        self.save_processed_image('edge.png', edge)

    def run_all_processing(self):
        self.apply_median_filter()
        self.apply_gaussian_filter()
        self.histogram_equalization()
        self.clahe()
        self.contrast_stretching(70, 0, 200, 255)
        self.canny_edge_detection(100, 200)