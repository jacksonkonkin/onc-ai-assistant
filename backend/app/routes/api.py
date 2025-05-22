from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "ONC Assistant backend is up!"}

@router.get("/status")
async def status():
    return {"status": "ok"}

@router.get("/health")
async def health_check():
    return {"health": "green"}