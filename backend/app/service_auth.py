import os
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
TOKEN_EXPIRE_MIN = int(os.getenv("TOKEN_EXPIRE_MIN", "1440"))



def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_pw: str, hashed_pw: str):
    return pwd_context.verify(plain_pw, hashed_pw)

def create_jwt(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt(token: str):
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def extract_token_from_header(authorization: str):
    """Extract token from Authorization header"""
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ")[1]
    return None

