from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

class ModelTraining:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(575, 575, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), input_shape=(575, 575, 3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation="sigmoid"))
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(self, training_data_dir, test_data_dir, epochs=5, batch_size=32):
        training_generate = ImageDataGenerator(
            rescale=1 / 255,
            rotation_range=7,
            horizontal_flip=True,
            shear_range=0.2,
            height_shift_range=0.07,
            zoom_range=0.2,
        )
        gerador_teste = ImageDataGenerator(rescale=1.0 / 255)

        training_base = training_generate.flow_from_directory(
            training_data_dir,
            target_size=(575, 575),
            batch_size=batch_size,
            class_mode="binary",
        )

        test_base = gerador_teste.flow_from_directory(
            test_data_dir,
            target_size=(575, 575),
            batch_size=batch_size,
            class_mode="binary",
        )

        self.model.fit(
            training_base,
            steps_per_epoch=len(training_base),
            epochs=epochs,
            validation_data=test_base,
            validation_steps=len(test_base),
        )

    def save_model(self, model_path="./models"):
        self.model.save(model_path)
        self.model.save_weights("./checkpoint.h5")
