import numpy as np
from keras.preprocessing import image
from keras import models
import cv2
import os
from model_training import ModelTraining
from image_processor import ImageProcessor

class ImagePredictor:
    def __init__(self):
        # Cria modelo e pesos
        classifier = ModelTraining()
        classifier.train("dataset/training_set", "dataset/test_set", epochs=5)
        classifier.save_model()

        # Carregar o modelo
        self.model = models.load_model("models")
        self.model.load_weights("checkpoint.h5")
    
    def get_classification(self, file_name):
        # Carregar a imagem usando o OpenCV
        img = cv2.imread(file_name)

        # Pré-processar a imagem
        img_resized = cv2.resize(img, (64, 64))
        img_rescaled = img_resized.astype(np.float32) / 255.0
        imageToPredict = np.expand_dims(img_rescaled, axis=0)

        # Realizar a classificação usando o modelo
        prediction = self.model.predict(imageToPredict)
        class_label = "Linfócito" if prediction > 0.5 else "Basófilo"
        return class_label


    def detect_and_contour_cell(self, file_name):   
        
        class_label = self.get_classification(file_name)

        image_processor = ImageProcessor(file_name)
        image_processor.run_all_processing()

        # Criar uma cópia da imagem para desenhar o contorno
        # img_with_contour = img.copy()

        # Converter a imagem para tons de cinza para detecção de contorno
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicar um desfoque para reduzir ruído
        # blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Aplicar a detecção de bordas usando o Canny
        # edges = cv2.Canny(blurred, 30, 150)

        # Encontrar os contornos na imagem
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Desenhar contorno da célula na imagem original
        # Substituir a cor (0, 255, 0) por (230, 230, 0)
        # cv2.drawContours(img_with_contour, contours, -1, (230, 230, 0), 2)

        # Obter o nome do arquivo de entrada sem a extensão
        file_name_without_extension = os.path.splitext(os.path.basename(file_name))[0]

        # Construir o caminho completo do arquivo de saída
        output_file = os.path.join(os.getcwd(), f"{file_name_without_extension}_com_contorno.jpg")

        # Salvar a imagem com o contorno
        cv2.imwrite(output_file, img_with_contour)

        return class_label, output_file