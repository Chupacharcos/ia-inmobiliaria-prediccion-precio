import pydantic
from pydantic import BaseModel

class InmobiliarioRequest(BaseModel):
    sqft_living: float
    bedrooms: float
    bathrooms: float
    floors: float
    condition: float
    grade: float
    yr_built: float
    yr_renovated: float
    lat: float
    long: float
    sqft_lot: float
    view: float
    waterfront: float

def validar_datos(request: InmobiliarioRequest):
    return request