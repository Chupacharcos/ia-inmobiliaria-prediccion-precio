import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/ml/predicciones", tags=["ML"])

class Prediccion(BaseModel):
    precio: float
    fecha: str

@router.get("/")
def get_predicciones():
    # Carga de datos
    datos = pd.read_csv("datos_predicciones.csv")
    # Bucle/filtro principal
    prediccion = datos[datos["fecha"] == "2022-01-01"]
    # Ordenación si aplica
    prediccion = prediccion.sort_values(by="precio")
    # Return final con el resultado completo
    return [{"precio": row["precio"], "fecha": row["fecha"]} for index, row in prediccion.iterrows()]