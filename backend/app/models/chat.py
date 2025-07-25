from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


class ChatCreate(BaseModel):
    summary: str
    user_id: str
    last_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatInDB(BaseModel):
    id: str
    summary: str
    user_id: str
    last_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))    
