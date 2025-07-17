from fastapi import APIRouter, HTTPException
from app.models.user import UserCreate
from app.db.mongo import users_collection
from app.service_auth import hash_password, verify_password, create_jwt
from bson import ObjectId

router = APIRouter()
DEFAULT_ONC_TOKEN = "your-dev-token"

@router.post("/signup")
async def signup(user: UserCreate):
    # Check if username exists
    existing = await users_collection.find_one({"username": user.username})
    if existing:
        raise HTTPException(400, "Username already exists.")

    hashed_pw = hash_password(user.password)
    token = user.onc_token or DEFAULT_ONC_TOKEN

    user_dict = {
        "username": user.username,
        "hashed_password": hashed_pw,
        "onc_token": token,
        "is_indigenous": user.is_indigenous,
        "role": user.role,
    }

    res = await users_collection.insert_one(user_dict)
    return {"message": "Signup successful", "id": str(res.inserted_id)}

@router.post("/login")
async def login(data: dict):
    username = data["username"]
    password = data["password"]
    
    user = await users_collection.find_one({"username": username})
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(401, "Invalid credentials")

    token = create_jwt({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}