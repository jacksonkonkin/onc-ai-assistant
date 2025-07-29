from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone

class MessageCreate(BaseModel):
    text: str
    chat_id: str
    user_id: str
    rating: Literal[-1, 0, 1]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MessageInDB(BaseModel):
    id: str
    text: str
    chat_id: str
    user_id: str
    rating: Literal[-1, 0, 1]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
