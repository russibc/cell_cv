class ModelTraining:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=128, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1, activation="sigmoid"))

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def train(self, training_data_dir, test_data_dir, epochs=5, batch_size=128):
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
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="binary",
        )

        self.test_base = gerador_teste.flow_from_directory(
            test_data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="binary",
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        self.model.fit(
            training_base,
            steps_per_epoch=len(training_base),
            epochs=epochs,
            validation_data=self.test_base,
            validation_steps=len(self.test_base),
            callbacks=[early_stopping]
        )

    def save_model(self, model_path="./models"):
        self.model.save(model_path)
        self.model.save_weights("./checkpoint.h5")