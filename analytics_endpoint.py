import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/analytics", tags=["Analytics"])

class Inmueble(BaseModel):
    id: int
    precio: float
    superficie: float

@router.get("/inmuebles")
def get_inmuebles():
    # Carga de datos
    datos = pd.read_csv("datos_inmuebles.csv")
    
    # Bucle/filtro principal
    inmuebles = []
    for index, row in datos.iterrows():
        inmueble = Inmueble(id=row["id"], precio=row["precio"], superficie=row["superficie"])
        inmuebles.append(inmueble)
    
    # Ordenación si aplica
    inmuebles.sort(key=lambda x: x.precio)
    
    # Return final con el resultado completo
    return inmuebles