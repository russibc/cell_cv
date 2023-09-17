import numpy as np
from keras.preprocessing import image
from keras import models
import cv2
from model_training import ModelTraining


class ImagePredictor:
    def __init__(self):
        # Cria modelo e pesos
        # classifier = ModelTraining()
        # classifier.train("dataset/training_set", "dataset/test_set", epochs=5)
        # classifier.save_model()

        # Carregar o modelo
        self.model = models.load_model("models")
        self.model.load_weights("checkpoint.h5")

    def get_classification(self, file_name):
        # Carregar a imagem usando o OpenCV
        img = cv2.imread(file_name)

        imagem_redimensionada = cv2.resize(img, (224, 224))

        # Pré-processar a imagem
        imageToPredict = np.expand_dims(imagem_redimensionada, axis=0)

        # Realizar a classificação usando o modelo
        prediction = self.model.predict(imageToPredict)

        class_index = np.argmax(prediction)

        class_labels = ['Linfocito', 'Basofilo']  # Substitua pelos seus rótulos
        class_label = class_labels[class_index]
        return class_label

    def detect_and_contour_cell(self, file_name):
        class_label = self.get_classification(file_name)
        return class_label
