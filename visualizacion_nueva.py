import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/visualizacion", tags=["Visualización"])

class DatosVisualizacion(BaseModel):
    fecha: str
    precio: float

@router.get("/datos")
def get_datos():
    # Carga de datos
    datos = pd.read_csv("datos_inmobiliarios.csv")
    
    # Filtro principal
    datos_filtrados = datos[datos["fecha"] > "2020-01-01"]
    
    # Ordenación
    datos_ordenados = datos_filtrados.sort_values(by="fecha")
    
    # Return final con el resultado completo
    return datos_ordenados.to_dict(orient="records")