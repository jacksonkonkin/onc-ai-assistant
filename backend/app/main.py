from fastapi import FastAPI
from app.routes import api

app = FastAPI(title="ONC AI Assistant Backend", version="1.0")

app.include_router(api.router)