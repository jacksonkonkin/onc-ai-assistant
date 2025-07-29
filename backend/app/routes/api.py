from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import sys
import os
from pathlib import Path
import logging

# Add the src directory to the path to import both feedback logger and pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from feedback.feedback_logger import SimpleFeedbackLogger, FeedbackRating

# Import the pipeline directly instead of using subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.api.pipeline import ONCPipeline

router = APIRouter()

# Initialize components
feedback_logger = SimpleFeedbackLogger()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline instance (will be setup on first use)
_pipeline_instance = None

def get_pipeline():
    """Get or initialize the pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing ONC Pipeline...")
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'onc_config.yaml')
            _pipeline_instance = ONCPipeline(config_path)
            
            # Setup pipeline with documents
            docs_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'onc_documents')
            success = _pipeline_instance.setup(docs_dir)
            
            if not success:
                logger.error("Pipeline setup failed")
                raise Exception("Pipeline setup failed")
            
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    return _pipeline_instance

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
        logger.info(f"🔍 Processing query: '{query.text}'")
        
        # Get list of existing CSV files before processing (use project root output directory)
        project_root = Path("/Users/jacksonkonkin/Documents/UVIC/SENG499/onc-ai-assistant")
        output_dir = project_root / "output"
        existing_files = set()
        if output_dir.exists():
            existing_files = set(f.name for f in output_dir.glob("*.csv"))
            logger.info(f"📁 Existing CSV files: {len(existing_files)} files")
        
        # Get pipeline instance and process query directly
        pipeline = get_pipeline()
        
        # DEBUG: Check current working directory
        logger.info(f"🔧 DEBUG: Backend current working directory: {os.getcwd()}")
        
        # Create query context with session info
        query_context = {
            'session_id': f"api_session_{query.chatID}_{query.userID}",
            'frontend_request': True
        }
        
        logger.info("=" * 80)
        logger.info("🔍 RAW PIPELINE OUTPUT:")
        
        # Process query directly through pipeline
        response = pipeline.query(query.text, query_context)
        
        logger.info(response)
        logger.info("=" * 80)
        
        # Check for CSV files to include in download response
        download_files = []
        
        # Check if this was a duplicate data response that mentions specific files or download IDs
        is_duplicate_response = "Similar Data Already Downloaded" in response and ("Files Created:" in response or "CSV files" in response)
        
        if is_duplicate_response:
            logger.info("🔄 Duplicate data detected with file references - including existing CSV files in response")
            # Include recent CSV files from all directories since this is duplicate data with file references
            import time
            recent_threshold = time.time() - 1800  # Files created in last 30 minutes for duplicates (more restrictive)
            
            # Check main output directory for recent files
            if output_dir.exists():
                for csv_file in output_dir.glob("*.csv"):
                    if csv_file.stat().st_mtime > recent_threshold:
                        file_info = {
                            "filename": csv_file.name,
                            "download_url": f"/download/output/{csv_file.name}",
                            "size": csv_file.stat().st_size
                        }
                        download_files.append(file_info)
                        logger.info(f"📄 Including existing file from main output: {csv_file.name} ({file_info['size']} bytes)")
            
            # Check backend output directory for recent files
            backend_output_dir = project_root / "backend" / "output"
            if backend_output_dir.exists():
                for csv_file in backend_output_dir.glob("*.csv"):
                    if csv_file.stat().st_mtime > recent_threshold:
                        # Check if we already found this file in main output
                        if not any(f["filename"] == csv_file.name for f in download_files):
                            file_info = {
                                "filename": csv_file.name,
                                "download_url": f"/download/backend/output/{csv_file.name}",
                                "size": csv_file.stat().st_size
                            }
                            download_files.append(file_info)
                            logger.info(f"📄 Including existing file from backend output: {csv_file.name} ({file_info['size']} bytes)")
        else:
            # Original logic for new files created during this request
            # Check main output directory
            if output_dir.exists():
                current_files = set(f.name for f in output_dir.glob("*.csv"))
                new_csv_files = current_files - existing_files
                
                logger.info(f"📊 New CSV files detected in main output: {len(new_csv_files)} files")
                
                for filename in new_csv_files:
                    file_path = output_dir / filename
                    if file_path.exists():
                        file_info = {
                            "filename": filename,
                            "download_url": f"/download/output/{filename}",
                            "size": file_path.stat().st_size
                        }
                        download_files.append(file_info)
                        logger.info(f"📄 New file in main output: {filename} ({file_info['size']} bytes)")
            
            # Also check backend output directory (where ONC package actually creates files)
            backend_output_dir = project_root / "backend" / "output"
            if backend_output_dir.exists():
                backend_files = set(f.name for f in backend_output_dir.glob("*.csv"))
                # Get files that are newer than when we started processing
                import time
                recent_threshold = time.time() - 300  # Files created in last 5 minutes
                
                for filename in backend_files:
                    file_path = backend_output_dir / filename
                    if file_path.exists() and file_path.stat().st_mtime > recent_threshold:
                        # Check if we already found this file in main output
                        if not any(f["filename"] == filename for f in download_files):
                            file_info = {
                                "filename": filename,
                                "download_url": f"/download/backend/output/{filename}",
                                "size": file_path.stat().st_size
                            }
                            download_files.append(file_info)
                            logger.info(f"📄 New file in backend output: {filename} ({file_info['size']} bytes)")
        
        # Include download links in response if files were found
        response_data = {"response": response}
        if download_files:
            response_data["downloads"] = download_files
            logger.info(f"✅ Response includes {len(download_files)} download links")
        else:
            logger.info("ℹ️ No download links in response")
            
        logger.info(f"📤 Final response length: {len(response)} characters")
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Pipeline error: {str(e)}")
        # Include more detailed error information for debugging
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Full error traceback: {error_details}")
        return {
            "error": "Internal server error",
            "details": str(e)
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

@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Serve CSV files for download"""
    # Security: Ensure file is in allowed directories
    project_root = "/Users/jacksonkonkin/Documents/UVIC/SENG499/onc-ai-assistant"
    
    # Construct full path - file_path already includes subdirectory (e.g., "output/filename.csv")
    full_path = os.path.join(project_root, file_path)
    
    # Security checks
    allowed_dirs = [
        os.path.join(project_root, "output"),
        os.path.join(project_root, "csv_downloads"),
        os.path.join(project_root, "backend", "output")  # Where ONC package actually creates files
    ]
    
    # Check if file exists and is in allowed directory
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security: ensure path is within allowed directories
    real_path = os.path.realpath(full_path)
    allowed = any(real_path.startswith(os.path.realpath(allowed_dir)) for allowed_dir in allowed_dirs)
    
    if not allowed:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(
        path=full_path,
        filename=os.path.basename(full_path),
        media_type='text/csv'
    )

@router.get("/files/list")
async def list_available_files():
    """List available CSV files for download"""
    files = []
    output_dirs = ["output", "csv_downloads"]
    
    for dir_name in output_dirs:
        dir_path = Path(f"/Users/jacksonkonkin/Documents/UVIC/SENG499/onc-ai-assistant/{dir_name}")
        if dir_path.exists():
            for csv_file in dir_path.glob("*.csv"):
                files.append({
                    "filename": csv_file.name,
                    "path": f"{dir_name}/{csv_file.name}",
                    "size": csv_file.stat().st_size,
                    "modified": csv_file.stat().st_mtime
                })
    
    return {"files": files}