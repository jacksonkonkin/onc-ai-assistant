from fastapi import APIRouter
from pydantic import BaseModel
import subprocess
import sys
import os

# Add the src directory to the path to import feedback logger
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from feedback.feedback_logger import SimpleFeedbackLogger, FeedbackRating

router = APIRouter()

# Initialize feedback logger
feedback_logger = SimpleFeedbackLogger()

class Query(BaseModel):
    text: str
    chatID: int | None = 0
    userID: int | None = 0

class FeedbackRequest(BaseModel):
    rating: str  # "thumbs_up" or "thumbs_down"
    query: str
    response: str
    metadata: dict | None = None

@router.get("/")
async def root():
    return {"message": "ONC Assistant backend is up!"}

@router.get("/status")
async def status():
    return {"status": "ok"}

@router.get("/health")
async def health_check():
    return {"health": "green"}

@router.post("/query")
async def query(query: Query):
    try:
        # Run the Python script with the query argument from project root
        result = subprocess.run(
            ["python", "onc_rag_pipeline_modular.py", "--query", query.text],
            capture_output=True,
            text=True,
            check=True,
            cwd="/Users/jacksonkonkin/Documents/UVIC/SENG499/onc-ai-assistant"
        )
        response = result.stdout.strip()
        response = response.split("Answer: ",1)[1]
        return {"response": response}
    except subprocess.CalledProcessError as e:
        return {
            "error": "Internal server error",
            "details": e.stderr.strip()
        }

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Convert string rating to enum
        if feedback.rating == "thumbs_up":
            rating_enum = FeedbackRating.THUMBS_UP
        elif feedback.rating == "thumbs_down":
            rating_enum = FeedbackRating.THUMBS_DOWN
        else:
            return {"error": "Invalid rating. Must be 'thumbs_up' or 'thumbs_down'"}
        
        # Log the feedback
        success = feedback_logger.log_feedback(
            rating=rating_enum,
            query=feedback.query,
            response=feedback.response,
            metadata=feedback.metadata
        )
        
        if success:
            return {"message": "Feedback logged successfully", "rating": feedback.rating}
        else:
            return {"error": "Failed to log feedback"}
            
    except Exception as e:
        return {"error": f"Error processing feedback: {str(e)}"}

@router.get("/feedback/stats")
async def get_feedback_stats():
    try:
        stats = feedback_logger.get_feedback_stats()
        return stats
    except Exception as e:
        return {"error": f"Error retrieving feedback stats: {str(e)}"}