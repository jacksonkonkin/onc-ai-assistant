from fastapi import APIRouter
from pydantic import BaseModel
import subprocess

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
    try:
        # Run the Python script with the query argument
        result = subprocess.run(
            ["python", "onc_rag_pipeline_modular.py", "--query", query.text],
            capture_output=True,
            text=True,
            check=True
        )
        response = result.stdout.strip()
        response = response.split("Answer: ",1)[1]
        return {"response": response}
    except subprocess.CalledProcessError as e:
        return {
            "error": "Internal server error",
            "details": e.stderr.strip()
        }