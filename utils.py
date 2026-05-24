import time
from fastapi import APIRouter

router = APIRouter()
@router.get('/utils')
def utils():
    return {'message': 'Utils endpoint'}