import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/visualizacion", tags=["Visualización"])
@router.get("/datos")
def get_datos():
    # Carga de datos
    datos = pd.read_csv("datos_inmobiliarios.csv")
    # Filtro principal
    datos_filtrados = datos[datos["precio"] > 0]
    # Ordenación
    datos_ordenados = datos_filtrados.sort_values(by="precio")
    # Return final
    return JSONResponse(content=datos_ordenados.to_dict(orient="records"), media_type="application/json")