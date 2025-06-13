import random

class RandomClassifier:
    def predict(self, x):
        # Zwraca losową etykietę jako wynik predykcji
        classes = ['cat', 'dog', 'bird']
        return [{"label": random.choice(classes), "probability": round(random.random(), 2)}]
