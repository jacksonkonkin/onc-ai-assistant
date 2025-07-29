from pydantic import BaseModel, Field
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    onc_token: Optional[str] = None
    is_indigenous: bool
    role: str  # Should validate one of the four

class UserInDB(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    onc_token: str
    is_indigenous: bool
    role: str
