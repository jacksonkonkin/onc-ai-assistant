from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Query(BaseModel):
    text: str
    chatID: int | None = 0
    userID: int | None = 0

@router.get("/")
async def root():
    return {"message": "ONC Assistant backend is up!"}

@router.get("/status")
async def status():
    return {"status": "ok"}

@router.get("/health")
async def health_check():
    return {"health": "green"}

@router.post("/query")
async def query(query: Query):
    # TODO: Process the text
    return {"response": f"Your input: {query.text}"}