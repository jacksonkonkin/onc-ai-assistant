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
import hashlib
import json

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
    
    def __init__(self, max_concurrent_downloads: int = 3, cleanup_hours: int = 24, session_id: str = None):
        """
        Initialize the download manager
        
        Args:
            max_concurrent_downloads: Maximum number of concurrent downloads
            cleanup_hours: Hours after which completed downloads are cleaned up
            session_id: Optional session ID for conversation-aware deduplication
        """
        self.active_downloads: Dict[str, Dict[str, Any]] = {}
        self.download_keys: Dict[str, str] = {}  # Maps download keys to download IDs for deduplication
        self.recent_downloads: Dict[str, Dict[str, Any]] = {}  # Cache of recent downloads for duplicate prevention
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_downloads,
            thread_name_prefix="bgdownload"
        )
        self.cleanup_hours = cleanup_hours
        self.recent_download_expiry_minutes = 30  # Prevent duplicates within 30 minutes
        self.lock = threading.RLock()  # Reentrant lock for nested locking
        
        logger.info(f"Background Download Manager initialized (max workers: {max_concurrent_downloads}, session: {self.session_id})")
    
    def _generate_download_key(self, params: Dict[str, Any]) -> str:
        """
        Generate enhanced unique key for download parameters to prevent duplicates
        
        Args:
            params: Download parameters
            
        Returns:
            Unique key string for these parameters
        """
        # Create normalized parameter dict for consistent hashing
        # For statistical queries, normalize resampling to enable data reuse
        resample_value = str(params.get('resample', 'none')).lower()
        # Treat minMaxAvg and none as equivalent for basic statistical operations
        # since minMaxAvg contains all the data needed for min, max, and avg
        if resample_value in ['minmaxavg', 'none']:
            resample_normalized = 'statistical_basic'  # Common key for statistical operations
        else:
            resample_normalized = resample_value
            
        normalized_params = {
            'location_code': str(params.get('location_code', '')).upper(),
            'device_category': str(params.get('device_category', '')).upper(),
            'date_from': str(params.get('date_from', '')),
            'date_to': str(params.get('date_to', '')),
            'resample': resample_normalized,
            'quality_control': str(params.get('quality_control', False)).lower()
        }
        
        # Sort keys for consistent ordering
        sorted_params = dict(sorted(normalized_params.items()))
        
        # Create hash for shorter, consistent keys
        param_string = json.dumps(sorted_params, sort_keys=True)
        key_hash = hashlib.md5(param_string.encode()).hexdigest()[:16]
        
        # Include session for conversation-aware deduplication
        return f"{self.session_id}:{key_hash}"
    
    def _generate_semantic_key(self, params: Dict[str, Any], user_query: str = None) -> str:
        """
        Generate semantic key for similar queries (broader deduplication)
        
        Args:
            params: Download parameters
            user_query: Original user query for semantic analysis
            
        Returns:
            Semantic key for similar requests
        """
        # Normalize date ranges to detect overlapping requests
        date_from = params.get('date_from', '')
        date_to = params.get('date_to', '')
        
        # For recent data requests, use a broader time window
        try:
            if date_from and date_to:
                from_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
                to_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
                
                # For monthly queries, include the month/year to prevent cross-month confusion
                duration = to_dt - from_dt
                if duration.days >= 25:  # Likely a monthly query
                    # Use month-year specificity for monthly queries
                    month_year = f"{from_dt.year}-{from_dt.month:02d}"
                    date_range_key = f"month_{month_year}"
                else:
                    # Use day range for shorter queries
                    from_day = from_dt.strftime('%Y-%m-%d')
                    to_day = to_dt.strftime('%Y-%m-%d')
                    date_range_key = f"{from_day}_{to_day}"
                
                semantic_params = {
                    'location': str(params.get('location_code', '')).upper(),
                    'device': str(params.get('device_category', '')).upper(),
                    'date_range': date_range_key,
                    'resample': str(params.get('resample', 'none')).lower(),
                    'session': self.session_id
                }
                
                semantic_string = json.dumps(semantic_params, sort_keys=True)
                return hashlib.md5(semantic_string.encode()).hexdigest()[:12]
        except Exception as e:
            logger.debug(f"Error generating semantic key: {e}")
        
        # Fallback to regular key generation
        return self._generate_download_key(params)[:12]
    
    def _check_recent_downloads(self, download_key: str, semantic_key: str) -> Optional[Dict[str, Any]]:
        """
        Check for recent downloads that might be duplicates
        
        Args:
            download_key: Exact download key
            semantic_key: Semantic key for similar requests
            
        Returns:
            Recent download info if found, None otherwise
        """
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(minutes=self.recent_download_expiry_minutes)
        
        # Clean up expired recent downloads first
        expired_keys = []
        for key, info in self.recent_downloads.items():
            if info['completion_time'] < cutoff_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.recent_downloads[key]
        
        # Check for exact match
        if download_key in self.recent_downloads:
            recent_info = self.recent_downloads[download_key]
            logger.info(f"Found exact recent download match: {download_key}")
            return recent_info
        
        # Check for semantic match
        for key, info in self.recent_downloads.items():
            if key.endswith(f":{semantic_key}") or semantic_key in key:
                logger.info(f"Found semantic recent download match: {key} -> {semantic_key}")
                return info
        
        return None
    
    def _add_to_recent_downloads(self, download_key: str, download_info: Dict[str, Any]):
        """
        Add completed download to recent downloads cache
        
        Args:
            download_key: Download key
            download_info: Download information
        """
        recent_info = {
            'download_id': download_info.get('download_id'),
            'completion_time': datetime.utcnow(),
            'status': download_info.get('status'),
            'files_created': download_info.get('files_created', []),
            'params': download_info.get('params', {}),
            'file_count': len(download_info.get('files_created', []))
        }
        
        # Store with both keys for better matching
        self.recent_downloads[download_key] = recent_info
        
        # Also store with semantic key if different
        try:
            semantic_key = self._generate_semantic_key(download_info.get('params', {}))
            if semantic_key != download_key:
                self.recent_downloads[f"semantic:{semantic_key}"] = recent_info
        except Exception as e:
            logger.debug(f"Error adding semantic key to recent downloads: {e}")
    
    def check_for_duplicate_request(self, download_params: Dict[str, Any], user_query: str = None) -> Optional[Dict[str, Any]]:
        """
        Check if this download request is a duplicate of recent downloads
        
        Args:
            download_params: Download parameters
            user_query: Original user query
            
        Returns:
            Recent download info if duplicate found, None otherwise
        """
        with self.lock:
            download_key = self._generate_download_key(download_params)
            semantic_key = self._generate_semantic_key(download_params, user_query)
            
            # Check active downloads first
            if download_key in self.download_keys:
                existing_download_id = self.download_keys[download_key]
                if existing_download_id in self.active_downloads:
                    existing_info = self.active_downloads[existing_download_id]
                    if existing_info['status'] in [DownloadStatus.PENDING, DownloadStatus.IN_PROGRESS]:
                        logger.info(f"Found active duplicate download: {existing_download_id}")
                        return {
                            'type': 'active_download',
                            'download_id': existing_download_id,
                            'status': existing_info['status'],
                            'start_time': existing_info['start_time']
                        }
            
            # Check recent completed downloads
            recent_download = self._check_recent_downloads(download_key, semantic_key)
            if recent_download:
                logger.info(f"Found recent completed download, preventing duplicate")
                return {
                    'type': 'recent_download',
                    'download_id': recent_download['download_id'],
                    'completion_time': recent_download['completion_time'],
                    'files_created': recent_download['files_created'],
                    'file_count': recent_download['file_count']
                }
            
            return None
    
    def start_background_download(self, download_params: Dict[str, Any], 
                                download_function: Callable,
                                callback: Optional[Callable] = None,
                                user_query: str = None,
                                force_download: bool = False) -> str:
        """
        Start a background download task with enhanced deduplication
        
        Args:
            download_params: Parameters for the download function
            download_function: Function that performs the actual download
            callback: Optional callback when download completes
            user_query: Original user query for semantic analysis
            force_download: Skip duplicate checking if True
            
        Returns:
            Unique download ID for tracking (existing ID if duplicate detected)
        """
        with self.lock:
            # Clean up old downloads before starting new one
            self._cleanup_old_downloads()
            
            # Check for duplicates unless forced
            if not force_download:
                duplicate_info = self.check_for_duplicate_request(download_params, user_query)
                if duplicate_info:
                    if duplicate_info['type'] == 'active_download':
                        logger.info(f"Returning existing active download: {duplicate_info['download_id']}")
                        return duplicate_info['download_id']
                    elif duplicate_info['type'] == 'recent_download':
                        logger.info(f"Found recent download with {duplicate_info['file_count']} files, skipping duplicate")
                        # Return special ID indicating recent download exists
                        return f"recent:{duplicate_info['download_id']}"
            
            download_key = self._generate_download_key(download_params)
            download_id = str(uuid.uuid4())
            
            # Check if we're at capacity
            active_count = sum(1 for d in self.active_downloads.values() 
                             if d['status'] in [DownloadStatus.PENDING, DownloadStatus.IN_PROGRESS])
            
            if active_count >= self.executor._max_workers:
                logger.warning(f"Download capacity reached. Queuing download {download_id}")
            
            # Create download tracking entry
            download_info = {
                'status': DownloadStatus.PENDING,
                'params': download_params.copy(),
                'download_key': download_key,
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
            self.download_keys[download_key] = download_id
            
            logger.info(f"Started background download {download_id} with key: {download_key}")
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
                            
                            # Add to recent downloads cache for duplicate prevention
                            try:
                                download_key = download_info.get('download_key')
                                if download_key:
                                    self._add_to_recent_downloads(download_key, {
                                        'download_id': download_id,
                                        'status': DownloadStatus.COMPLETED,
                                        'files_created': download_info.get('files_created', []),
                                        'params': download_info.get('params', {})
                                    })
                            except Exception as e:
                                logger.warning(f"Failed to add download to recent cache: {e}")
                        else:
                            download_info['status'] = DownloadStatus.FAILED
                            download_info['error'] = result.get('message', 'Unknown error')
                            logger.error(f"Download {download_id} failed: {download_info['error']}")
                    else:
                        download_info['status'] = DownloadStatus.COMPLETED
                        logger.info(f"Download {download_id} completed")
                        
                        # Add to recent downloads cache for duplicate prevention
                        try:
                            download_key = download_info.get('download_key')
                            if download_key:
                                self._add_to_recent_downloads(download_key, {
                                    'download_id': download_id,
                                    'status': DownloadStatus.COMPLETED,
                                    'files_created': download_info.get('files_created', []),
                                    'params': download_info.get('params', {})
                                })
                        except Exception as e:
                            logger.warning(f"Failed to add download to recent cache: {e}")
                    
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
            # Clean up download_keys mapping
            download_info = self.active_downloads[download_id]
            download_key = download_info.get('download_key')
            if download_key and download_key in self.download_keys:
                if self.download_keys[download_key] == download_id:
                    del self.download_keys[download_key]
            
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
        
        # Clear all tracking data
        with self.lock:
            self.active_downloads.clear()
            self.download_keys.clear()
            
        logger.info("Background Download Manager shutdown complete")


# Global instance for use across the application
_global_download_manager: Optional[BackgroundDownloadManager] = None


def get_download_manager(session_id: str = None) -> BackgroundDownloadManager:
    """
    Get the global download manager instance
    
    Args:
        session_id: Optional session ID for conversation-aware deduplication
    
    Returns:
        Global BackgroundDownloadManager instance
    """
    global _global_download_manager
    if _global_download_manager is None:
        _global_download_manager = BackgroundDownloadManager(session_id=session_id)
    return _global_download_manager


def shutdown_download_manager():
    """Shutdown the global download manager"""
    global _global_download_manager
    if _global_download_manager is not None:
        _global_download_manager.shutdown()
        _global_download_manager = None