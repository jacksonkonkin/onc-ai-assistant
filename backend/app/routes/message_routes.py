from fastapi import APIRouter, HTTPException, Path, Body
from app.models.message import MessageCreate, MessageInDB
from app.db.mongo import messages_collection, chats_collection
from datetime import datetime, timezone
from bson import ObjectId
from app.routes.chat_history import update_chat

router = APIRouter()
SUMMARY_LIMIT = 100

# gets a single message given a message ID.
@router.get("/api/messages/{message_id}")
async def get_message(message_id: str = Path(...)):
    message = await messages_collection.find_one({"_id": ObjectId(message_id)})
    if not message:
        raise HTTPException(404, "Message not found")
    return {
        "id": str(message["_id"]),
        "chat_id": message["chat_id"],
        "user_id": message["user_id"],
        "text": message["text"],
        "rating": message["rating"],
        "timestamp": message["timestamp"]
    }

@router.get("/api/messages")
async def get_messages_for_chat(chat_id: str):
    messages = messages_collection.find({"chat_id": chat_id}).sort("timestamp", 1)
    result = []
    async for msg in messages:
        result.append({
            "id": str(msg["_id"]),
            "text": msg["text"],
            "user_id": msg["user_id"],
            "rating": msg["rating"],
            "timestamp": msg["timestamp"]
        })
    return result

# updates the rating of a single message
@router.put("/api/messages/{message_id}/rating")
async def update_message_rating(message_id: str, rating: int = Body(..., embed=True)):
    if rating not in [-1, 0, 1]:
        raise HTTPException(400, "Rating must be -1, 0, or 1")
    result = await messages_collection.update_one(
        {"_id": ObjectId(message_id)},
        {"$set": {"rating": rating}}
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Message not found")
    return {"message": "Rating updated"}

# creates a single message given text, chat id, and user id (NOTE: The response from our backend should use -1 as their user id)
@router.post("/api/messages")
async def create_message(data: dict = Body(...)):
    required_keys = {"text", "chat_id", "user_id"}
    if not required_keys.issubset(data.keys()):
        raise HTTPException(400, f"Missing one of: {required_keys}")
    
    message_data = {
        "chat_id": data["chat_id"],
        "user_id": data["user_id"],
        "text": data["text"],
        "rating": 0,
        "timestamp": datetime.now(timezone.utc)
    }
    result = await messages_collection.insert_one(message_data)

    summary = data["text"][:100] + ("..." if len(data["text"]) > 100 else "")
    await update_chat(
        chat_id=data["chat_id"],
        updates={
            "summary": summary,
            "last_timestamp": datetime.now(timezone.utc)
        }
    )

    return {"id": str(result.inserted_id), "message": "Message created"}

# Deletes a message given a message id.
@router.delete("/api/messages/{message_id}")
async def delete_message(message_id: str = Path(...)):
    result = await messages_collection.delete_one({"_id": ObjectId(message_id)})
    if result.deleted_count == 0:
        raise HTTPException(404, "Message not found")
    return {"message": "Message deleted"}

@router.get("/api/messages-by-rating")
async def get_messages_by_rating(rating: int):
    if rating not in [-1, 0, 1]:
        raise HTTPException(400, detail="Rating must be -1, 0, or 1")

    messages = messages_collection.find({"rating": rating}).sort("timestamp", 1)
    result = []
    async for msg in messages:
        result.append({
            "id": str(msg["_id"]),
            "text": msg["text"],
            "user_id": msg["user_id"],
            "chat_id": msg["chat_id"],
            "rating": msg["rating"],
            "timestamp": msg["timestamp"]
        })
    return result