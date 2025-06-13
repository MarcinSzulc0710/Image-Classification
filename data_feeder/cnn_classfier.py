import tensorflow as tf
from tensorflow.keras import layers, models

class SimpleCNNClassifier:
    def __init__(self):
        self.model = models.Sequential([
            layers.Input(shape=(64, 64, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, x):
        preds = self.model.predict(x)
        return preds.tolist()
