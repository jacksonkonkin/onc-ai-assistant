from fastapi import APIRouter, HTTPException, Path, Body
from typing import List
from app.models.chat import ChatCreate, ChatInDB
from app.db.mongo import chats_collection, messages_collection
from bson import ObjectId
from datetime import datetime, timezone

router = APIRouter()

# return the list of all chats associated with a user, along with their summary and last timestamp.
@router.get("/api/chat-histories")
async def get_user_chats(user_id: str):
    chats = chats_collection.find({"user_id": user_id}).sort("last_timestamp", -1)
    result = []
    async for chat in chats:
        result.append({
            "id": str(chat["_id"]),
            "summary": chat["summary"],
            "last_timestamp": chat.get("last_timestamp")
        })
    return result


@router.post("/api/chat-histories")
async def create_chat(chat: ChatCreate):
    chat_dict = chat.dict()
    chat_dict["last_timestamp"] = datetime.now(timezone.utc)
    result = await chats_collection.insert_one(chat_dict)
    return {"id": str(result.inserted_id), "message": "Chat created"}


@router.put("/api/chat-histories/{chat_id}")
async def update_chat(
    chat_id: str = Path(...),
    updates: dict = Body(...)
):
    allowed_fields = {"summary", "last_timestamp"}
    update_data = {k: v for k, v in updates.items() if k in allowed_fields}
    if "last_timestamp" in update_data:
        # Optionally convert to datetime if not already
        if isinstance(update_data["last_timestamp"], str):
            update_data["last_timestamp"] = datetime.fromisoformat(update_data["last_timestamp"])
    result = await chats_collection.update_one(
        {"_id": ObjectId(chat_id)},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Chat not found")
    return {"message": "Chat updated"}


@router.delete("/api/chat-histories/{chat_id}")
async def delete_chat(chat_id: str = Path(...)):
    chat_obj_id = ObjectId(chat_id)
    await messages_collection.delete_many({"chat_id": chat_id})
    result = await chats_collection.delete_one({"_id": chat_obj_id})
    if result.deleted_count == 0:
        raise HTTPException(404, "Chat not found")
    return {"message": "Chat and related messages deleted"}
