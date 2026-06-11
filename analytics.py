import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/analytics", tags=["Análisis"])

class Inmueble(BaseModel):
    sqft_living: float
    sqft_lot: float
    bedrooms: int
    bathrooms: int

@router.get("/inmuebles")
def get_inmuebles():
    # Carga de datos
    datos = pd.read_csv("datos_inmuebles.csv")
    
    # Filtro principal
    inmuebles = datos[(datos["sqft_living"] > 1000) & (datos["sqft_living"] < 2000)]
    
    # Ordenación
    inmuebles = inmuebles.sort_values(by="sqft_living")
    
    # Return final
    return inmuebles.to_dict(orient="records")