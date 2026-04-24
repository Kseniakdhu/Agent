import joblib
import numpy as np
import pandas as pd


class IrisModel:
    def __init__(self, model_path='iris_model.pkl'):
        self.model = joblib.load(model_path)
        self.trained_feature_names = None
        if hasattr(self.model, "feature_names_in_"):
            self.trained_feature_names = list(self.model.feature_names_in_)

    def predict(self, features):
        arr = np.array(features).reshape(1, -1)

        if self.trained_feature_names:
            df = pd.DataFrame(arr, columns=self.trained_feature_names)
            prediction = self.model.predict(df)
        else:
            prediction = self.model.predict(arr)

        names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        label = int(prediction[0])
        species = names[label]
        return f'Вид ириса: {species}'