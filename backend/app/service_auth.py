from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "super-secret"  # Change to env var in prod
ALGORITHM = "HS256"
TOKEN_EXPIRE_MIN = 60 * 24

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_pw: str, hashed_pw: str):
    return pwd_context.verify(plain_pw, hashed_pw)

def create_jwt(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
