import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class PrecioInmobiliario:
    def __init__(self):
        self.modelo = RandomForestRegressor()
    def entrenar(self, datos):
        X = datos.drop('precio', axis=1)
        y = datos['precio']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.modelo.fit(X_train, y_train)
    def predecir(self, datos):
        return self.modelo.predict(datos)