import numpy as np
from keras.preprocessing import image
from keras import models
import cv2

class ImagePredictor:
    def __init__(self):
        # Cria modelo e pesos
        self.classifier = ModelTraining()
        self.classifier.train("dataset/training_set", "dataset/test_set")
        self.classifier.save_model()

        # Carregar o modelo
        self.model = models.load_model("models")
        self.model.load_weights("checkpoint.h5")

    def get_model(self):
      return self.model

    def get_classification(self, file_name):
      prediction = self.get_prediction(file_name)
      return "Linfocito" if prediction > 0.5 else "Basofilo"

    def get_prediction(self, file_name):
        # Carregar a imagem usando o OpenCV
        img = cv2.imread(file_name)

        # Redimensionar a imagem
        imagem_redimensionada = cv2.resize(img, (224, 224))

        # Normalizar a imagem para o intervalo [0, 1]
        imagem_normalizada = imagem_redimensionada / 255.0

        # Pré-processar a imagem
        image_to_predict = np.expand_dims(imagem_normalizada, axis=0)

        # Realizar a classificação usando o modelo
        prediction = self.model.predict(image_to_predict)

        prediction = (prediction > 0.5).astype(int)

        return prediction