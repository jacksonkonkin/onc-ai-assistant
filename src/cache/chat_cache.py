import logging
import json
import ast
from typing import List
from src.cache.redis_client import redis_client

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY_LENGTH = 50
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 hours

class CachingFunc:
    #role must be user or assistant
    @staticmethod
    def save_chat(user_id: str, role: str, message: str):
        try:
            if role not in ("user", "assistant"):
                raise ValueError("Invalid role. Must be 'user' or 'assistant'.")
            entry = {"role": role, "message": message}
            key = f"chat:{user_id}"
            redis_client.rpush(key, json.dumps(entry))
            redis_client.ltrim(key, -MAX_HISTORY_LENGTH, -1)
            redis_client.expire(key, SESSION_TTL_SECONDS)
        except Exception as e:
            logger.error(f"Error saving chat for user_id={user_id}: {e}")

    @staticmethod
    def get_chat_history(user_id: str) -> str:
        try:
            key = f"chat:{user_id}"
            raw_history = redis_client.lrange(key, 0, -1)
            history = [ast.literal_eval(entry) for entry in raw_history]
            return json.dumps(history, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error getting chat history for user_id={user_id}: {e}")
            return "[]"

    @staticmethod
    def delete_session_chats(user_id: str):
        try:
            redis_client.delete(f"chat:{user_id}")
        except Exception as e:
            logger.error(f"Error deleting chat history for user_id={user_id}: {e}")



"""
def main():
    user_id = "tytyty"
    #CachingFunc.save_chat(user_id, "user", "Hello AI!")
    #CachingFunc.save_chat(user_id, "assistant", "Hi there! How can I help you today?")

    chat = CachingFunc.get_chat_history(user_id)
    print(chat)

    CachingFunc.delete_session_chats(user_id)

if __name__ == "__main__":
    main()

    
"""