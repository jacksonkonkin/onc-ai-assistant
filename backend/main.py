from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import api
from app.routes import auth

origins = [
    "http://localhost:3000",
]

app = FastAPI(title="ONC AI Assistant Backend", version="1.0")
app.add_middleware(
       CORSMiddleware,
       allow_origins=origins,
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )

app.include_router(api.router)
app.include_router(auth.router)