import pydantic
from pydantic import BaseModel, Field

class InmobiliarioRequest(BaseModel):
    sqft_living: float = Field(default=1800, ge=100, le=20000, description="Superficie habitable (sqft)")
    bedrooms: float = Field(default=3, ge=0, le=20, description="Numero de habitaciones")
    bathrooms: float = Field(default=2, ge=0, le=15, description="Numero de banos")
    floors: float = Field(default=1, ge=1, le=4, description="Numero de plantas")
    condition: float = Field(default=3, ge=1, le=5, description="Estado de conservacion (1-5)")
    grade: float = Field(default=7, ge=1, le=13, description="Calidad de construccion (1-13)")
    yr_built: float = Field(default=1990, ge=1800, le=2015, description="Ano de construccion")
    yr_renovated: float = Field(default=0, ge=0, le=2015, description="Ano de renovacion (0 = nunca)")
    lat: float = Field(default=47.5, ge=-90.0, le=90.0, description="Latitud")
    long: float = Field(default=-122.2, ge=-180.0, le=180.0, description="Longitud")
    sqft_lot: float = Field(default=7500, ge=100, le=1000000, description="Superficie de la parcela (sqft)")
    view: float = Field(default=0, ge=0, le=4, description="Calidad de las vistas (0-4)")
    waterfront: float = Field(default=0, ge=0, le=4, description="Calidad del agua (0-4)")