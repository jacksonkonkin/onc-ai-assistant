
from fastapi import APIRouter, HTTPException, Header
from app.models.user import UserCreate
from app.db.mongo import users_collection
from app.service_auth import hash_password, verify_password, create_jwt, verify_jwt, extract_token_from_header

from bson import ObjectId

router = APIRouter()
DEFAULT_ONC_TOKEN = "your-dev-token"

@router.post("/signup")
async def signup(user: UserCreate):
    # Check if username exists
    existing_user = await users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(400, "Username already exists.")
    
    # Check if email exists
    existing_email = await users_collection.find_one({"email": user.email})
    if existing_email:
        raise HTTPException(400, "Email already exists.")

    hashed_pw = hash_password(user.password)
    token = user.onc_token or DEFAULT_ONC_TOKEN

    user_dict = {
        "username": user.username,
        "email": user.email,
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

# Development endpoint to view users (admin only)
@router.get("/users")
async def get_all_users(authorization: str = Header(None)):
    # Verify authentication
    if not authorization:
        raise HTTPException(401, "Authorization header required")
    
    token = extract_token_from_header(authorization)
    if not token:
        raise HTTPException(401, "Invalid authorization format")
    
    payload = verify_jwt(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    
    # Check if user has admin role
    user_role = payload.get("role")
    if user_role != "admin":
        raise HTTPException(403, "Admin access required")
    
    users = []
    async for user in users_collection.find():
        # Remove sensitive data for display
        user_data = {
            "id": str(user["_id"]),
            "username": user.get("username", "N/A"),
            "email": user.get("email", "Not provided"),
            "role": user.get("role", "N/A"),
            "is_indigenous": user.get("is_indigenous", False),
            "onc_token": user.get("onc_token", "Not provided")
        }
        users.append(user_data)
    return {"users": users, "total": len(users)}

# Admin-only endpoint to get user statistics
@router.get("/admin/stats")
async def get_admin_stats(authorization: str = Header(None)):
    # Verify authentication
    if not authorization:
        raise HTTPException(401, "Authorization header required")
    
    token = extract_token_from_header(authorization)
    if not token:
        raise HTTPException(401, "Invalid authorization format")
    
    payload = verify_jwt(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    
    # Check if user has admin role
    user_role = payload.get("role")
    if user_role != "admin":
        raise HTTPException(403, "Admin access required")
    
    total_users = await users_collection.count_documents({})
    admin_users = await users_collection.count_documents({"role": "admin"})
    regular_users = total_users - admin_users
    
    # Count by role
    role_counts = {}
    async for user in users_collection.find():
        role = user.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    return {
        "total_users": total_users,
        "admin_users": admin_users,
        "regular_users": regular_users,
        "role_breakdown": role_counts
    }

    

