import time
from fastapi import APIRouter

router = APIRouter(prefix="/inmobiliario")
@router.get('/precio')
def get_precio():
    return {}