import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/visualizacion-precio", tags=["Visualización de Precios"])

class PrecioInmueble(BaseModel):
    precio: float
    ubicacion: str

@router.get("/precio-inmueble")
def get_precio_inmueble():
    # Carga de datos
    datos = pd.read_csv("datos_inmuebles.csv")
    
    # Filtro principal
    datos_filtrados = datos[datos["precio"] > 0]
    
    # Ordenación
    datos_ordenados = datos_filtrados.sort_values(by="precio")
    
    # Return final
    return datos_ordenados.to_dict(orient="records")