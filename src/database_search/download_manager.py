"""
Background Download Manager
Manages concurrent data downloads while providing immediate user responses
"""

import uuid
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import concurrent.futures
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class DownloadStatus:
    """Download status constants"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"  
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class BackgroundDownloadManager:
    """
    Manages background downloads while providing immediate responses
    """
    
    def __init__(self, max_concurrent_downloads: int = 3, cleanup_hours: int = 24):
        """
        Initialize the download manager
        
        Args:
            max_concurrent_downloads: Maximum number of concurrent downloads
            cleanup_hours: Hours after which completed downloads are cleaned up
        """
        self.active_downloads: Dict[str, Dict[str, Any]] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_downloads,
            thread_name_prefix="bgdownload"
        )
        self.cleanup_hours = cleanup_hours
        self.lock = threading.RLock()  # Reentrant lock for nested locking
        
        logger.info(f"Background Download Manager initialized (max workers: {max_concurrent_downloads})")
    
    def start_background_download(self, download_params: Dict[str, Any], 
                                download_function: Callable,
                                callback: Optional[Callable] = None) -> str:
        """
        Start a background download task
        
        Args:
            download_params: Parameters for the download function
            download_function: Function that performs the actual download
            callback: Optional callback when download completes
            
        Returns:
            Unique download ID for tracking
        """
        download_id = str(uuid.uuid4())
        
        with self.lock:
            # Clean up old downloads before starting new one
            self._cleanup_old_downloads()
            
            # Check if we're at capacity
            active_count = sum(1 for d in self.active_downloads.values() 
                             if d['status'] in [DownloadStatus.PENDING, DownloadStatus.IN_PROGRESS])
            
            if active_count >= self.executor._max_workers:
                logger.warning(f"Download capacity reached. Queuing download {download_id}")
            
            # Create download tracking entry
            download_info = {
                'status': DownloadStatus.PENDING,
                'params': download_params.copy(),
                'start_time': datetime.utcnow(),
                'end_time': None,
                'result': None,
                'error': None,
                'files_created': [],
                'callback': callback,
                'future': None
            }
            
            # Submit the download task
            future = self.executor.submit(self._download_wrapper, download_id, download_function, download_params)
            download_info['future'] = future
            download_info['status'] = DownloadStatus.IN_PROGRESS
            
            self.active_downloads[download_id] = download_info
            
            logger.info(f"Started background download {download_id}")
            return download_id
    
    def _download_wrapper(self, download_id: str, download_function: Callable, params: Dict[str, Any]):
        """
        Wrapper function that executes the download and handles completion
        """
        try:
            logger.info(f"Executing background download {download_id}")
            
            # Execute the actual download in an isolated manner
            try:
                result = download_function(**params)
                logger.info(f"Download function completed for {download_id}")
            except Exception as download_error:
                logger.error(f"Download function failed for {download_id}: {download_error}")
                result = {
                    'status': 'error',
                    'message': f"Download function failed: {str(download_error)}",
                    'csv_files': []
                }
            
            with self.lock:
                if download_id in self.active_downloads:
                    download_info = self.active_downloads[download_id]
                    download_info['end_time'] = datetime.utcnow()
                    download_info['result'] = result
                    
                    # Extract file paths if available
                    if isinstance(result, dict):
                        csv_files = result.get('csv_files', [])
                        if csv_files:
                            download_info['files_created'] = csv_files
                        
                        # Update status based on result
                        if result.get('status') == 'success':
                            download_info['status'] = DownloadStatus.COMPLETED
                            logger.info(f"Download {download_id} completed successfully")
                        else:
                            download_info['status'] = DownloadStatus.FAILED
                            download_info['error'] = result.get('message', 'Unknown error')
                            logger.error(f"Download {download_id} failed: {download_info['error']}")
                    else:
                        download_info['status'] = DownloadStatus.COMPLETED
                        logger.info(f"Download {download_id} completed")
                    
                    # Execute callback if provided (in a non-blocking way)
                    callback = download_info.get('callback')
                    if callback:
                        try:
                            # Execute callback without blocking the main thread
                            import threading
                            callback_thread = threading.Thread(
                                target=self._safe_callback_execution,
                                args=(callback, download_id, result),
                                daemon=True
                            )
                            callback_thread.start()
                        except Exception as e:
                            logger.error(f"Failed to start callback thread for {download_id}: {e}")
            
        except Exception as e:
            logger.error(f"Download {download_id} failed with exception: {e}")
            
            with self.lock:
                if download_id in self.active_downloads:
                    download_info = self.active_downloads[download_id]
                    download_info['status'] = DownloadStatus.FAILED
                    download_info['error'] = str(e)
                    download_info['end_time'] = datetime.utcnow()
    
    def _safe_callback_execution(self, callback: Callable, download_id: str, result: Dict[str, Any]):
        """
        Safely execute download completion callback in a separate thread
        
        Args:
            callback: Callback function to execute
            download_id: Download ID
            result: Download result
        """
        try:
            callback(download_id, result)
            logger.debug(f"Callback executed successfully for download {download_id}")
        except Exception as e:
            logger.error(f"Download callback failed for {download_id}: {e}")
    
    def get_download_status(self, download_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of a download
        
        Args:
            download_id: Download ID to check
            
        Returns:
            Download status information or None if not found
        """
        with self.lock:
            if download_id not in self.active_downloads:
                return None
            
            download_info = self.active_downloads[download_id].copy()
            
            # Remove internal fields
            if 'future' in download_info:
                del download_info['future']
            if 'callback' in download_info:
                del download_info['callback']
            
            # Add computed fields
            if download_info['start_time']:
                download_info['elapsed_seconds'] = (
                    (download_info['end_time'] or datetime.utcnow()) - download_info['start_time']
                ).total_seconds()
            
            # Add file information
            files = download_info.get('files_created', [])
            download_info['file_count'] = len(files)
            
            if files:
                # Get file sizes
                file_details = []
                total_size = 0
                for file_path in files:
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path)
                        total_size += size
                        file_details.append({
                            'path': file_path,
                            'filename': os.path.basename(file_path),
                            'size_bytes': size,
                            'size_mb': round(size / (1024 * 1024), 2)
                        })
                
                download_info['files'] = file_details
                download_info['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            return download_info
    
    def get_download_files(self, download_id: str) -> List[str]:
        """
        Get list of files created by a download
        
        Args:
            download_id: Download ID
            
        Returns:
            List of file paths created by the download
        """
        with self.lock:
            if download_id not in self.active_downloads:
                return []
            
            return self.active_downloads[download_id].get('files_created', [])
    
    def cancel_download(self, download_id: str) -> bool:
        """
        Cancel a download if it's still running
        
        Args:
            download_id: Download ID to cancel
            
        Returns:
            True if cancellation was attempted, False if download not found
        """
        with self.lock:
            if download_id not in self.active_downloads:
                return False
            
            download_info = self.active_downloads[download_id]
            future = download_info.get('future')
            
            if future and not future.done():
                cancelled = future.cancel()
                if cancelled:
                    download_info['status'] = DownloadStatus.FAILED
                    download_info['error'] = "Cancelled by user"
                    download_info['end_time'] = datetime.utcnow()
                    logger.info(f"Cancelled download {download_id}")
                return cancelled
            
            return False
    
    def list_active_downloads(self) -> List[str]:
        """
        Get list of all active download IDs
        
        Returns:
            List of download IDs that are pending or in progress
        """
        with self.lock:
            active_statuses = [DownloadStatus.PENDING, DownloadStatus.IN_PROGRESS]
            return [
                download_id for download_id, info in self.active_downloads.items()
                if info['status'] in active_statuses
            ]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of download manager state
        
        Returns:
            Summary information about downloads
        """
        with self.lock:
            status_counts = {}
            for info in self.active_downloads.values():
                status = info['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            active_count = status_counts.get(DownloadStatus.IN_PROGRESS, 0)
            pending_count = status_counts.get(DownloadStatus.PENDING, 0)
            
            return {
                'total_downloads': len(self.active_downloads),
                'active_downloads': active_count,
                'pending_downloads': pending_count,
                'completed_downloads': status_counts.get(DownloadStatus.COMPLETED, 0),
                'failed_downloads': status_counts.get(DownloadStatus.FAILED, 0),
                'max_workers': self.executor._max_workers,
                'status_breakdown': status_counts
            }
    
    def _cleanup_old_downloads(self):
        """Clean up completed downloads older than cleanup_hours"""
        if not self.cleanup_hours:
            return
            
        cutoff_time = datetime.utcnow() - timedelta(hours=self.cleanup_hours)
        to_remove = []
        
        for download_id, info in self.active_downloads.items():
            end_time = info.get('end_time')
            status = info.get('status')
            
            # Only clean up completed/failed downloads
            if status in [DownloadStatus.COMPLETED, DownloadStatus.FAILED] and end_time:
                if end_time < cutoff_time:
                    to_remove.append(download_id)
        
        for download_id in to_remove:
            del self.active_downloads[download_id]
            logger.debug(f"Cleaned up old download {download_id}")
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old downloads")
    
    def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shutdown the download manager
        
        Args:
            wait: Whether to wait for downloads to complete
            timeout: Maximum time to wait for shutdown
        """
        logger.info("Shutting down Background Download Manager")
        
        if wait:
            # Cancel all pending downloads
            with self.lock:
                for download_id, info in self.active_downloads.items():
                    if info['status'] == DownloadStatus.PENDING:
                        future = info.get('future')
                        if future:
                            future.cancel()
        
        self.executor.shutdown(wait=wait, timeout=timeout)
        logger.info("Background Download Manager shutdown complete")


# Global instance for use across the application
_global_download_manager: Optional[BackgroundDownloadManager] = None


def get_download_manager() -> BackgroundDownloadManager:
    """
    Get the global download manager instance
    
    Returns:
        Global BackgroundDownloadManager instance
    """
    global _global_download_manager
    if _global_download_manager is None:
        _global_download_manager = BackgroundDownloadManager()
    return _global_download_manager


def shutdown_download_manager():
    """Shutdown the global download manager"""
    global _global_download_manager
    if _global_download_manager is not None:
        _global_download_manager.shutdown()
        _global_download_manager = None