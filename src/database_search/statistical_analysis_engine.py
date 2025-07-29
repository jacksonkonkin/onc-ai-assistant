"""
Statistical Analysis Engine for Oceanographic Data
Provides advanced statistical analysis capabilities for sensor data queries
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logging.warning("Matplotlib/Seaborn not available - plotting disabled")

from .advanced_data_downloader import AdvancedDataDownloader
from .enhanced_parameter_extractor import EnhancedParameterExtractor
from .statistical_visualizer import StatisticalVisualizer

logger = logging.getLogger(__name__)


class StatisticalAnalysisEngine:
    """
    Advanced statistical analysis engine for oceanographic data
    Handles complex queries like min/max/average calculations, trends, and correlations
    """
    
    def __init__(self, onc_token: str = None, data_downloader: AdvancedDataDownloader = None, session_id: str = None):
        """
        Initialize the statistical analysis engine
        
        Args:
            onc_token: ONC API token
            data_downloader: Existing data downloader instance (optional)
            session_id: Session ID for conversation-aware duplicate prevention
        """
        self.data_downloader = data_downloader or AdvancedDataDownloader(onc_token)
        self.parameter_extractor = EnhancedParameterExtractor()
        self.visualizer = StatisticalVisualizer()
        self.session_id = session_id
        
        # Statistical operation mappings
        self.statistical_operations = {
            'min': self._calculate_minimum,
            'minimum': self._calculate_minimum,
            'max': self._calculate_maximum,
            'maximum': self._calculate_maximum,
            'avg': self._calculate_average,
            'average': self._calculate_average,
            'mean': self._calculate_average,
            'sum': self._calculate_sum,
            'total': self._calculate_sum,
            'std': self._calculate_standard_deviation,
            'stdev': self._calculate_standard_deviation,
            'var': self._calculate_variance,
            'variance': self._calculate_variance,
            'median': self._calculate_median,
            'mode': self._calculate_mode,
            'range': self._calculate_range,
            'count': self._calculate_count,
            'trend': self._calculate_trend,
            'correlation': self._calculate_correlation,
            'seasonal': self._calculate_seasonal_patterns
        }
        
        # Time window patterns for aggregation
        self.time_windows = {
            'hourly': '1H',
            'daily': '1D', 
            'weekly': '1W',
            'monthly': '1M',
            'yearly': '1Y',
            'minute': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T'
        }
        
        logger.info("Statistical Analysis Engine initialized")
    
    def _check_and_cleanup_active_downloads(self):
        """
        Check for and cleanup active downloads that might interfere with new statistical requests
        """
        try:
            from .download_manager import get_download_manager
            download_manager = get_download_manager()
            
            active_downloads = download_manager.list_active_downloads()
            if active_downloads:
                logger.warning(f"Found {len(active_downloads)} active downloads during statistical query")
                
                # Get summary for more details
                summary = download_manager.get_summary()
                logger.info(f"Download manager summary: {summary}")
                
                # If there are too many active downloads, this might indicate a problem
                if len(active_downloads) > 2:
                    logger.warning(f"Excessive active downloads ({len(active_downloads)}), may indicate download loop issue")
                    
                    # Cancel downloads that have been running too long (> 10 minutes)
                    import time
                    current_time = time.time()
                    for download_id in active_downloads:
                        download_info = download_manager.get_download_status(download_id)
                        if download_info:
                            elapsed = download_info.get('elapsed_seconds', 0)
                            if elapsed > 600:  # 10 minutes
                                logger.warning(f"Cancelling long-running download {download_id} (elapsed: {elapsed}s)")
                                download_manager.cancel_download(download_id)
                
                # Brief wait to let any active downloads settle
                import time
                time.sleep(1)
                
        except Exception as e:
            logger.warning(f"Error checking active downloads: {e}")
            # Don't fail the statistical query if download checking fails
    
    def process_statistical_query(self, query: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Process a statistical query and return computed results
        
        Args:
            query: Natural language statistical query
            include_metadata: Whether to include processing metadata
            
        Returns:
            Dictionary with statistical results and metadata
        """
        start_time = datetime.now()
        logger.info(f"Processing statistical query: '{query}'")
        
        try:
            # Step 1: Extract statistical parameters from query
            stats_params = self._extract_statistical_parameters(query)
            if not stats_params['success']:
                return {
                    'status': 'error',
                    'stage': 'parameter_extraction',
                    'message': stats_params['message'],
                    'data': None
                }
            
            # Step 2: Get the statistical data using ONC API's aggregation
            data_params = stats_params['data_parameters']
            # Add statistical operations to data_params for API calls
            data_params['statistical_operations'] = stats_params['statistical_parameters']['operations']
            raw_data_result = self._get_raw_data_for_analysis(data_params, query)
            
            if raw_data_result['status'] != 'success':
                return raw_data_result
            
            # Step 3: Perform statistical analysis
            analysis_result = self._perform_statistical_analysis(
                raw_data_result['data'], 
                stats_params['statistical_parameters']
            )
            
            # Step 4: Format results for presentation
            formatted_result = self._format_statistical_results(
                analysis_result, 
                stats_params, 
                query
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Step 5: Build final response
            response = {
                'status': 'success',
                'query': query,
                'statistical_results': formatted_result,
                'raw_data_summary': {
                    'total_records': len(raw_data_result.get('data', [])),
                    'time_range': raw_data_result.get('time_range'),
                    'parameters_analyzed': stats_params['data_parameters']
                }
            }
            
            if include_metadata:
                response['metadata'] = {
                    'processing_time': round(total_time, 2),
                    'statistical_operations': stats_params['statistical_parameters']['operations'],
                    'time_aggregation': stats_params['statistical_parameters'].get('time_window'),
                    'data_quality_score': self._calculate_aggregated_data_quality_score(raw_data_result['data'])
                }
            
            logger.info(f"Statistical query completed in {total_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing statistical query: {e}")
            return {
                'status': 'error',
                'stage': 'statistical_processing',
                'message': f"Statistical analysis failed: {str(e)}",
                'data': None
            }
    
    def _extract_statistical_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract statistical parameters from natural language query
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with extracted parameters
        """
        try:
            # Use enhanced parameter extractor for basic parameters
            basic_extraction = self.parameter_extractor.extract_parameters(query)
            
            if basic_extraction['status'] != 'success':
                return {
                    'success': False,
                    'message': basic_extraction['message']
                }
            
            # Extract statistical-specific parameters
            query_lower = query.lower()
            
            # Detect statistical operations
            operations = []
            for op_keyword, op_function in self.statistical_operations.items():
                if op_keyword in query_lower:
                    operations.append(op_keyword)
            
            if not operations:
                # Default to basic statistics if none specified
                operations = ['average', 'min', 'max']
            
            # Detect time aggregation windows
            time_window = None
            for window_name, pandas_freq in self.time_windows.items():
                if window_name in query_lower:
                    time_window = {'name': window_name, 'freq': pandas_freq}
                    break
            
            # Detect comparison parameters
            comparison_params = self._extract_comparison_parameters(query_lower)
            
            # Detect grouping parameters
            grouping_params = self._extract_grouping_parameters(query_lower)
            
            return {
                'success': True,
                'data_parameters': basic_extraction['parameters'],
                'statistical_parameters': {
                    'operations': operations,
                    'time_window': time_window,
                    'comparison': comparison_params,
                    'grouping': grouping_params,
                    'original_query': query
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting statistical parameters: {e}")
            return {
                'success': False,
                'message': f"Parameter extraction failed: {str(e)}"
            }
    
    def _extract_comparison_parameters(self, query_lower: str) -> Dict[str, Any]:
        """Extract comparison parameters from query"""
        comparison_params = {
            'compare_to_previous': False,
            'compare_locations': False,
            'compare_devices': False,
            'threshold_analysis': None
        }
        
        # Detect comparison keywords
        if any(phrase in query_lower for phrase in ['compared to', 'vs', 'versus', 'difference']):
            comparison_params['compare_to_previous'] = True
        
        if any(phrase in query_lower for phrase in ['between locations', 'across sites', 'location comparison']):
            comparison_params['compare_locations'] = True
        
        if any(phrase in query_lower for phrase in ['between devices', 'device comparison', 'sensor comparison']):
            comparison_params['compare_devices'] = True
        
        # Detect threshold analysis
        threshold_keywords = ['above', 'below', 'greater than', 'less than', 'exceeds', 'under']
        for keyword in threshold_keywords:
            if keyword in query_lower:
                comparison_params['threshold_analysis'] = keyword
                break
        
        return comparison_params
    
    def _extract_grouping_parameters(self, query_lower: str) -> Dict[str, Any]:
        """Extract grouping parameters from query"""
        grouping_params = {
            'group_by_time': None,
            'group_by_location': False,
            'group_by_device': False,
            'group_by_depth': False
        }
        
        # Detect time-based grouping
        time_groupings = ['by hour', 'by day', 'by week', 'by month', 'by year', 'hourly', 'daily', 'weekly', 'monthly', 'yearly']
        for grouping in time_groupings:
            if grouping in query_lower:
                grouping_params['group_by_time'] = grouping.replace('by ', '').replace('ly', '')
                break
        
        # Detect other grouping criteria
        if 'by location' in query_lower or 'per location' in query_lower:
            grouping_params['group_by_location'] = True
        
        if 'by device' in query_lower or 'per device' in query_lower:
            grouping_params['group_by_device'] = True
        
        if 'by depth' in query_lower or 'per depth' in query_lower:
            grouping_params['group_by_depth'] = True
        
        return grouping_params
    
    def _parse_onc_csv_file(self, csv_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse ONC CSV file format, skipping header comments and extracting data
        
        Args:
            csv_file_path: Path to the ONC CSV file
            
        Returns:
            Dictionary with parsed data and metadata
        """
        try:
            logger.info(f"Starting to parse ONC CSV file: {csv_file_path}")
            # Read the file and find where data starts (after ## END HEADER)
            with open(csv_file_path, 'r') as f:
                lines = f.readlines()
            
            logger.info(f"Read {len(lines)} lines from CSV file")
            data_start_idx = None
            header_info = {}
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Extract key metadata from header
                if line.startswith('#TOTSAMPLE:'):
                    # Parse the format: #TOTSAMPLE: 576                                  / Samples in File
                    try:
                        sample_part = line.split(':')[1].strip()
                        # Split by '/' and take the first part, then strip whitespace
                        sample_count = sample_part.split('/')[0].strip()
                        header_info['total_samples'] = int(sample_count)
                        logger.info(f"Found total samples: {header_info['total_samples']}")
                    except (ValueError, IndexError):
                        header_info['total_samples'] = 0
                elif line.startswith('#DATEFROM:'):
                    header_info['date_from'] = line.split(':', 1)[1].strip()
                elif line.startswith('#DATETO:'):
                    header_info['date_to'] = line.split(':', 1)[1].strip()
                elif line.startswith('#RESAMPTYP:'):
                    header_info['resample_type'] = line.split(':', 1)[1].strip().strip('"')
                elif line.startswith('#RESAMPPRD:'):
                    header_info['resample_period'] = line.split(':', 1)[1].strip().strip('"')
                elif line.startswith('#DEPTH:'):
                    # Extract depth information: #DEPTH:         9.0                              / Depth (m)
                    try:
                        depth_part = line.split(':')[1].strip()
                        depth_value = depth_part.split('/')[0].strip()
                        header_info['depth'] = float(depth_value)
                    except (ValueError, IndexError):
                        header_info['depth'] = None
                
                # Find where actual data starts
                if line == '## END HEADER' or (line and not line.startswith('#')):
                    data_start_idx = i + 1 if line == '## END HEADER' else i
                    break
            
            if data_start_idx is None:
                logger.warning(f"Could not find data section in {csv_file_path}")
                return None
                
            logger.info(f"Found data starting at line {data_start_idx}")
            
            # Find the header line (contains column names in quotes)
            header_line_idx = None
            for i in range(len(lines)):
                if lines[i].strip().startswith('#"Time UTC'):
                    header_line_idx = i
                    break
            
            if header_line_idx is not None:
                logger.info(f"Found header line at index {header_line_idx}")
                # Extract column names from the header line
                header_line = lines[header_line_idx].strip()
                # Remove the # and parse the quoted column names
                header_line = header_line[1:]  # Remove #
                
                # Handle complex CSV parsing with proper quote handling
                import csv
                from io import StringIO
                header_buffer = StringIO(header_line)
                csv_reader = csv.reader(header_buffer, delimiter=',', quotechar='"')
                try:
                    column_names = next(csv_reader)
                    # Clean up column names
                    column_names = [col.strip() for col in column_names]
                    logger.info(f"Successfully parsed {len(column_names)} column names")
                except Exception as e:
                    logger.warning(f"Failed to parse CSV header with csv.reader: {e}")
                    # Fallback to simple parsing
                    column_names = [col.strip().strip('"') for col in header_line.split('", "')]
                    # Fix the first and last column names
                    if column_names:
                        column_names[0] = column_names[0].strip('"')
                        column_names[-1] = column_names[-1].strip('"')
                    logger.info(f"Fallback parsing produced {len(column_names)} column names")
            else:
                logger.warning("No header line found in CSV file")
                column_names = None
            
            # Read the CSV data starting from the data section
            data_lines = lines[data_start_idx:]
            
            # Create a temporary string buffer with just the data
            from io import StringIO
            data_buffer = StringIO(''.join(data_lines))
            
            # Read into pandas DataFrame with proper column names and CSV parsing
            try:
                logger.info(f"Attempting to parse {len(data_lines)} data lines with pandas")
                # Use pandas CSV parser with proper handling for complex CSV format
                df = pd.read_csv(data_buffer, sep=',', header=None, names=column_names, 
                               quotechar='"', skipinitialspace=True, on_bad_lines='skip')
                logger.info(f"Successfully parsed DataFrame with shape: {df.shape}")
            except Exception as e:
                logger.warning(f"Failed to parse CSV data with pandas: {e}")
                # Try with more lenient parsing
                data_buffer.seek(0)  # Reset buffer position
                df = pd.read_csv(data_buffer, sep=',', header=None, names=column_names, 
                               on_bad_lines='skip', engine='python')
                logger.info(f"Fallback parsing produced DataFrame with shape: {df.shape}")
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            logger.info(f"Final DataFrame shape: {df.shape}, columns: {len(df.columns)}")
            logger.info(f"Sample column names: {list(df.columns)[:5]}")
            
            return {
                'dataframe': df,
                'metadata': header_info,
                'total_records': len(df),
                'columns': list(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error parsing ONC CSV file {csv_file_path}: {e}")
            return None
    
    def _get_raw_data_for_analysis(self, data_params: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """
        Get statistical data using ONC API's native aggregation capabilities
        
        Args:
            data_params: Data parameters extracted from query
            query: Original user query for duplicate detection
            
        Returns:
            Statistical data result dictionary
        """
        try:
            # Check for active downloads that might conflict
            self._check_and_cleanup_active_downloads()
            
            # Use ONC API's native statistical aggregation
            stats_operations = data_params.get('statistical_operations', ['average'])
            results = {}
            
            # Determine what type of statistical data we need
            needs_basic_stats = any(op in ['min', 'minimum', 'max', 'maximum', 'avg', 'average', 'mean'] 
                                   for op in stats_operations)
            needs_other_stats = any(op not in ['min', 'minimum', 'max', 'maximum', 'avg', 'average', 'mean'] 
                                   for op in stats_operations)
            
            # Use minMaxAvg to get all three basic statistics in one download
            if needs_basic_stats:
                download_params = {
                    'location_code': data_params['location_code'],
                    'device_category': data_params.get('device_category', 'CTD'),
                    'date_from': data_params['start_time'],
                    'date_to': data_params['end_time'],
                    'output_dir': 'output',
                    'quality_control': True,
                    'resample': 'minMaxAvg'  # Get min, max, and average in one download
                }
                
                logger.info("Downloading minMaxAvg data for comprehensive statistics")
                
                # Add timeout protection to prevent infinite downloads
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Statistical download timeout after 300 seconds")
                
                # Set timeout for 5 minutes
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)  # 5 minutes timeout
                
                try:
                    download_result = self.data_downloader.download_csv_data(
                        **download_params,
                        session_id=self.session_id,
                        user_query=query[:100]  # Pass truncated query for duplicate detection
                    )
                except TimeoutError as timeout_error:
                    logger.error(f"Statistical analysis download timed out: {timeout_error}")
                    return {
                        'status': 'error',
                        'message': 'Statistical data download timed out after 5 minutes. This may indicate network issues or high server load.',
                        'data': []
                    }
                finally:
                    signal.alarm(0)  # Cancel the alarm
                
                if download_result['status'] in ['success', 'duplicate_recent']:
                    csv_files = download_result.get('csv_files', [])
                    if download_result['status'] == 'duplicate_recent':
                        logger.info(f"Using {len(csv_files)} recently downloaded CSV files for minMaxAvg processing")
                    else:
                        logger.info(f"Downloaded {len(csv_files)} CSV files for minMaxAvg processing")
                    
                    if csv_files:
                        # Process the minMaxAvg CSV - it contains all three statistics
                        for csv_file in csv_files:
                            try:
                                logger.info(f"Processing minMaxAvg CSV file: {csv_file}")
                                # Read CSV with header processing for ONC format
                                aggregated_data = self._parse_onc_csv_file(csv_file)
                                logger.info(f"Parse result: {aggregated_data is not None}")
                                if aggregated_data is not None:
                                    logger.info(f"Aggregated data keys: {list(aggregated_data.keys()) if isinstance(aggregated_data, dict) else 'Not a dict'}")
                                    if 'dataframe' in aggregated_data:
                                        logger.info(f"DataFrame shape: {aggregated_data['dataframe'].shape}")
                                        logger.info(f"DataFrame empty: {aggregated_data['dataframe'].empty}")
                                        logger.info(f"DataFrame columns: {list(aggregated_data['dataframe'].columns)[:10]}")  # First 10 columns
                                
                                if aggregated_data is not None and 'dataframe' in aggregated_data and not aggregated_data['dataframe'].empty:
                                    logger.info(f"Successfully parsed CSV data with {len(aggregated_data['dataframe'])} rows")
                                    # Extract all operations from the minMaxAvg data
                                    for operation in stats_operations:
                                        if operation in ['min', 'minimum', 'max', 'maximum', 'avg', 'average', 'mean']:
                                            results[operation] = {
                                                'data': aggregated_data,
                                                'csv_file': csv_file,
                                                'operation': operation,
                                                'resample_type': 'minMaxAvg'
                                            }
                                            logger.info(f"Added operation '{operation}' to results")
                                else:
                                    if aggregated_data is None:
                                        logger.warning(f"CSV file {csv_file} parsing returned None")
                                    elif 'dataframe' not in aggregated_data:
                                        logger.warning(f"CSV file {csv_file} parsing did not return dataframe")
                                    elif aggregated_data['dataframe'].empty:
                                        logger.warning(f"CSV file {csv_file} produced empty dataframe")
                                    else:
                                        logger.warning(f"CSV file {csv_file} produced unknown issue")
                            except Exception as e:
                                logger.error(f"Could not process minMaxAvg CSV file {csv_file}: {e}")
                    else:
                        logger.warning("No CSV files returned from minMaxAvg download")
                elif download_result['status'] == 'no_data':
                    logger.warning(f"No data available for minMaxAvg operation: {download_result.get('message', 'No data found')}")
                    # Don't continue processing other operations if no data exists
                    return {
                        'status': 'error',
                        'message': f"No data available for the requested time period: {download_result.get('date_range', 'Unknown range')}",
                        'data': []
                    }
                elif download_result['status'] == 'duplicate_active':
                    logger.info("Active download detected for minMaxAvg processing, waiting...")
                    return {
                        'status': 'error',
                        'message': 'Data download is currently in progress for similar request. Please try again in a few moments.',
                        'data': []
                    }
                else:
                    logger.error(f"minMaxAvg download failed: {download_result.get('message', 'Unknown error')}")
                    # Don't continue with infinite retries - return error immediately
                    return {
                        'status': 'error',
                        'message': f"Failed to download minMaxAvg data: {download_result.get('message', 'Download failed')}",
                        'data': []
                    }
            
            # Handle other statistical operations that need raw data or special processing
            if needs_other_stats:
                other_operations = [op for op in stats_operations 
                                  if op not in ['min', 'minimum', 'max', 'maximum', 'avg', 'average', 'mean']]
                
                for operation in other_operations:
                    # These operations need raw data for local computation
                    download_params = {
                        'location_code': data_params['location_code'],
                        'device_category': data_params.get('device_category', 'CTD'),
                        'date_from': data_params['start_time'],
                        'date_to': data_params['end_time'],
                        'output_dir': 'output',
                        'quality_control': True,
                        'resample': 'none'  # Raw data for complex statistical operations
                    }
                    
                    # Add timeout protection for other statistical operations
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)  # 5 minutes timeout
                    
                    try:
                        download_result = self.data_downloader.download_csv_data(
                            **download_params,
                            session_id=self.session_id,
                            user_query=f"{operation} for {query[:80]}"  # Pass operation context for duplicate detection
                        )
                    except TimeoutError as timeout_error:
                        logger.error(f"Statistical analysis download timed out for operation {operation}: {timeout_error}")
                        continue  # Skip this operation
                    finally:
                        signal.alarm(0)  # Cancel the alarm
                    
                    if download_result['status'] in ['success', 'duplicate_recent']:
                        csv_files = download_result.get('csv_files', [])
                        if download_result['status'] == 'duplicate_recent':
                            logger.info(f"Using {len(csv_files)} recently downloaded CSV files for {operation}")
                        if csv_files:
                            for csv_file in csv_files:
                                try:
                                    raw_data = self._parse_onc_csv_file(csv_file)
                                    if raw_data:
                                        results[operation] = {
                                            'data': raw_data,
                                            'csv_file': csv_file,
                                            'operation': operation,
                                            'resample_type': 'none'
                                        }
                                except Exception as e:
                                    logger.warning(f"Could not process raw CSV file {csv_file}: {e}")
                    elif download_result['status'] == 'no_data':
                        logger.warning(f"No data available for operation {operation}: {download_result.get('message', 'No data found')}")
                        continue  # Skip this operation, no data available
                    elif download_result['status'] == 'duplicate_active':
                        logger.info(f"Active download detected for {operation}, skipping...")
                        continue  # Skip this operation, try others
            
            if not results:
                return {
                    'status': 'error',
                    'message': 'No statistical data could be retrieved from ONC API',
                    'data': []
                }
            
            return {
                'status': 'success',
                'message': f'Retrieved {len(results)} statistical aggregations from ONC API',
                'data': results,
                'time_range': f"{data_params['start_time']} to {data_params['end_time']}",
                'operations': list(results.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting statistical data from ONC API: {e}")
            return {
                'status': 'error',
                'message': f"Failed to get statistical data: {str(e)}",
                'data': []
            }
    
    def _perform_statistical_analysis(self, data: Dict[str, Any], stats_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process statistical data from ONC API's aggregated results
        
        Args:
            data: Dictionary with aggregated data from ONC API
            stats_params: Statistical parameters
            
        Returns:
            Analysis results dictionary
        """
        try:
            results = {
                'operations_performed': [],
                'statistics': {},
                'time_series_analysis': {},
                'quality_metrics': {}
            }
            
            # Process each statistical operation's data
            for operation, operation_data in data.items():
                if 'data' in operation_data and operation_data['data']:
                    parsed_data = operation_data['data']
                    df = parsed_data['dataframe']
                    metadata = parsed_data['metadata']
                    
                    logger.info(f"Processing operation: {operation}")
                    logger.info(f"DataFrame shape: {df.shape}")
                    logger.info(f"DataFrame columns: {list(df.columns)}")
                    
                    # Identify numeric columns for analysis
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_columns) > 0:
                        # Extract statistical values from the aggregated DataFrame
                        stats_result = {}
                        
                        for col in numeric_columns:
                            if col in df.columns and len(df[col].dropna()) > 0:
                                # For aggregated data, we typically get one or few values
                                if operation in ['average', 'mean', 'avg']:
                                    avg_value = df[col].mean()
                                    stats_result[col] = {
                                        'value': float(avg_value) if pd.notna(avg_value) else None,
                                        'count': int(df[col].count()),
                                        'unit': self._get_column_unit(col),
                                        'metadata': metadata
                                    }
                                
                                elif operation in ['min', 'minimum']:
                                    # For minMax resampled data, extract from the correct column
                                    if operation in ['min', 'minimum'] and 'temperature' in col.lower() and 'min' in col.lower() and 'value' in col.lower():
                                        # This is the Temperature Min Value column - use the min of these values
                                        min_value = df[col].min()
                                        min_idx = df[col].idxmin()
                                        stats_result[col] = {
                                            'value': float(min_value) if pd.notna(min_value) else None,
                                            'timestamp': str(min_idx) if pd.notna(min_idx) else None,
                                            'unit': self._get_column_unit(col),
                                            'metadata': metadata,
                                            'resampling_note': 'Extracted from Temperature Min Value column'
                                        }
                                
                                elif operation in ['max', 'maximum']:
                                    # For minMax resampled data, extract from the correct column
                                    if operation in ['max', 'maximum'] and 'temperature' in col.lower() and 'max' in col.lower() and 'value' in col.lower():
                                        # This is the Temperature Max Value column - use the max of these values
                                        max_value = df[col].max()
                                        max_idx = df[col].idxmax()
                                        stats_result[col] = {
                                            'value': float(max_value) if pd.notna(max_value) else None,
                                            'timestamp': str(max_idx) if pd.notna(max_idx) else None,
                                            'unit': self._get_column_unit(col),
                                            'metadata': metadata,
                                            'resampling_note': 'Extracted from Temperature Max Value column'
                                        }
                        
                        if stats_result:
                            results['statistics'][operation] = stats_result
                            results['operations_performed'].append(operation)
                    
                    # Add quality metrics based on metadata
                    if metadata:
                        results['quality_metrics'][operation] = {
                            'total_samples': metadata.get('total_samples', 0),
                            'date_range': f"{metadata.get('date_from', '')} to {metadata.get('date_to', '')}",
                            'resample_type': metadata.get('resample_type', 'Unknown'),
                            'resample_period': metadata.get('resample_period', 'Unknown')
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing statistical analysis: {e}")
            return {
                'error': f"Statistical analysis failed: {str(e)}",
                'operations_performed': [],
                'statistics': {},
                'time_series_analysis': {},
                'quality_metrics': {}
            }
    
    # Statistical operation implementations
    def _calculate_minimum(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate minimum values"""
        result = {}
        for col in columns:
            # Check for minMaxAvg format columns first (e.g., temperature_min)
            min_col = f"{col}_min"
            if min_col in data.columns:
                min_val = data[min_col].iloc[0] if not data[min_col].empty else None
                result[col] = {
                    'value': float(min_val) if pd.notna(min_val) else None,
                    'timestamp': str(data.index[0]) if not data.empty else None,
                    'unit': self._get_column_unit(col)
                }
            # Fallback to raw data calculation
            elif col in data.columns:
                min_val = data[col].min()
                min_idx = data[col].idxmin()
                result[col] = {
                    'value': float(min_val) if pd.notna(min_val) else None,
                    'timestamp': str(min_idx) if pd.notna(min_idx) else None,
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_maximum(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate maximum values"""
        result = {}
        for col in columns:
            # Check for minMaxAvg format columns first (e.g., temperature_max)
            max_col = f"{col}_max"
            if max_col in data.columns:
                max_val = data[max_col].iloc[0] if not data[max_col].empty else None
                result[col] = {
                    'value': float(max_val) if pd.notna(max_val) else None,
                    'timestamp': str(data.index[0]) if not data.empty else None,
                    'unit': self._get_column_unit(col)
                }
            # Fallback to raw data calculation
            elif col in data.columns:
                max_val = data[col].max()
                max_idx = data[col].idxmax()
                result[col] = {
                    'value': float(max_val) if pd.notna(max_val) else None,
                    'timestamp': str(max_idx) if pd.notna(max_idx) else None,
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_average(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate average values"""
        result = {}
        for col in columns:
            # Check for minMaxAvg format columns first (e.g., temperature_avg or temperature_mean)
            avg_col = f"{col}_avg"
            mean_col = f"{col}_mean"
            if avg_col in data.columns:
                avg_val = data[avg_col].iloc[0] if not data[avg_col].empty else None
                result[col] = {
                    'value': float(avg_val) if pd.notna(avg_val) else None,
                    'count': len(data) if not data.empty else 0,
                    'unit': self._get_column_unit(col)
                }
            elif mean_col in data.columns:
                avg_val = data[mean_col].iloc[0] if not data[mean_col].empty else None
                result[col] = {
                    'value': float(avg_val) if pd.notna(avg_val) else None,
                    'count': len(data) if not data.empty else 0,
                    'unit': self._get_column_unit(col)
                }
            # Fallback to raw data calculation
            elif col in data.columns:
                avg_val = data[col].mean()
                result[col] = {
                    'value': float(avg_val) if pd.notna(avg_val) else None,
                    'count': int(data[col].count()),
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_sum(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sum values"""
        result = {}
        for col in columns:
            if col in data.columns:
                sum_val = data[col].sum()
                result[col] = {
                    'value': float(sum_val) if pd.notna(sum_val) else None,
                    'count': int(data[col].count()),
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_standard_deviation(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate standard deviation"""
        result = {}
        for col in columns:
            if col in data.columns:
                std_val = data[col].std()
                result[col] = {
                    'value': float(std_val) if pd.notna(std_val) else None,
                    'mean': float(data[col].mean()) if pd.notna(data[col].mean()) else None,
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_variance(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate variance"""
        result = {}
        for col in columns:
            if col in data.columns:
                var_val = data[col].var()
                result[col] = {
                    'value': float(var_val) if pd.notna(var_val) else None,
                    'unit': f"{self._get_column_unit(col)}²"
                }
        return result
    
    def _calculate_median(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate median values"""
        result = {}
        for col in columns:
            if col in data.columns:
                median_val = data[col].median()
                result[col] = {
                    'value': float(median_val) if pd.notna(median_val) else None,
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_mode(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate mode values"""
        result = {}
        for col in columns:
            if col in data.columns:
                mode_val = data[col].mode()
                result[col] = {
                    'value': float(mode_val.iloc[0]) if len(mode_val) > 0 else None,
                    'frequency': int(data[col].value_counts().iloc[0]) if len(data[col].value_counts()) > 0 else 0,
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_range(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate range (max - min)"""
        result = {}
        for col in columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                range_val = max_val - min_val if pd.notna(min_val) and pd.notna(max_val) else None
                result[col] = {
                    'value': float(range_val) if range_val is not None else None,
                    'min': float(min_val) if pd.notna(min_val) else None,
                    'max': float(max_val) if pd.notna(max_val) else None,
                    'unit': self._get_column_unit(col)
                }
        return result
    
    def _calculate_count(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate count of non-null values"""
        result = {}
        for col in columns:
            if col in data.columns:
                result[col] = {
                    'value': int(data[col].count()),
                    'total_rows': len(data),
                    'null_count': int(data[col].isnull().sum()),
                    'completeness_percentage': round((data[col].count() / len(data)) * 100, 2)
                }
        return result
    
    def _calculate_trend(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate trend analysis"""
        result = {}
        for col in columns:
            if col in data.columns and len(data) > 1:
                # Simple linear trend
                x = np.arange(len(data))
                y = data[col].dropna()
                if len(y) > 1:
                    x_clean = x[:len(y)]
                    slope, intercept = np.polyfit(x_clean, y, 1)
                    
                    result[col] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                        'correlation_coefficient': float(np.corrcoef(x_clean, y)[0, 1]) if len(y) > 1 else 0,
                        'unit': f"{self._get_column_unit(col)}/sample"
                    }
        return result
    
    def _calculate_correlation(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation matrix"""
        if len(columns) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        correlation_matrix = data[columns].corr()
        
        result = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': []
        }
        
        # Find strong correlations (>0.7 or <-0.7)
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:
                        result['strong_correlations'].append({
                            'parameter1': col1,
                            'parameter2': col2,
                            'correlation': float(corr),
                            'strength': 'strong positive' if corr > 0.7 else 'strong negative'
                        })
        
        return result
    
    def _calculate_seasonal_patterns(self, data: pd.DataFrame, columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate seasonal patterns in data"""
        result = {}
        
        if not hasattr(data.index, 'month'):
            return {'error': 'Need datetime index for seasonal analysis'}
        
        for col in columns:
            if col in data.columns:
                # Group by month
                monthly_stats = data.groupby(data.index.month)[col].agg(['mean', 'std', 'count'])
                
                result[col] = {
                    'monthly_averages': monthly_stats['mean'].to_dict(),
                    'monthly_std': monthly_stats['std'].to_dict(),
                    'monthly_counts': monthly_stats['count'].to_dict(),
                    'peak_month': int(monthly_stats['mean'].idxmax()),
                    'low_month': int(monthly_stats['mean'].idxmin()),
                    'seasonal_variation': float(monthly_stats['mean'].max() - monthly_stats['mean'].min()),
                    'unit': self._get_column_unit(col)
                }
        
        return result
    
    def _perform_time_aggregation(self, data: pd.DataFrame, time_window: Dict[str, str], 
                                 columns: List[str]) -> Dict[str, Any]:
        """Perform time-based aggregation"""
        freq = time_window['freq']
        
        # Resample data according to time window
        aggregated = data[columns].resample(freq).agg(['mean', 'min', 'max', 'std', 'count'])
        
        result = {
            'time_window': time_window['name'],
            'aggregation_frequency': freq,
            'time_periods': len(aggregated),
            'aggregated_data': {}
        }
        
        for col in columns:
            if col in aggregated.columns.get_level_values(0):
                result['aggregated_data'][col] = {
                    'mean': aggregated[(col, 'mean')].dropna().to_dict(),
                    'min': aggregated[(col, 'min')].dropna().to_dict(),
                    'max': aggregated[(col, 'max')].dropna().to_dict(),
                    'std': aggregated[(col, 'std')].dropna().to_dict(),
                    'count': aggregated[(col, 'count')].dropna().to_dict()
                }
        
        return result
    
    def _calculate_data_quality_metrics(self, data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        quality_metrics = {
            'total_records': len(data),
            'columns_analyzed': len(columns),
            'completeness': {},
            'outliers': {},
            'time_coverage': {}
        }
        
        # Completeness analysis
        for col in columns:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                quality_metrics['completeness'][col] = {
                    'completeness_percentage': round((1 - null_count / len(data)) * 100, 2),
                    'missing_values': int(null_count),
                    'valid_values': int(len(data) - null_count)
                }
        
        # Simple outlier detection using IQR method
        for col in columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_threshold_low = Q1 - 1.5 * IQR
                outlier_threshold_high = Q3 + 1.5 * IQR
                
                outliers = data[(data[col] < outlier_threshold_low) | (data[col] > outlier_threshold_high)]
                
                quality_metrics['outliers'][col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': round(len(outliers) / len(data) * 100, 2),
                    'threshold_low': float(outlier_threshold_low),
                    'threshold_high': float(outlier_threshold_high)
                }
        
        # Time coverage analysis
        if hasattr(data.index, 'to_pydatetime'):
            time_range = data.index.max() - data.index.min()
            expected_samples = None  # Would need to know sampling frequency
            
            quality_metrics['time_coverage'] = {
                'start_time': str(data.index.min()),
                'end_time': str(data.index.max()),
                'duration_days': time_range.days,
                'duration_hours': round(time_range.total_seconds() / 3600, 2)
            }
        
        return quality_metrics
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        if data.empty:
            return 0.0
        
        # Simple quality score based on completeness
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            return 0.0
        
        total_completeness = 0
        for col in numeric_columns:
            completeness = 1 - (data[col].isnull().sum() / len(data))
            total_completeness += completeness
        
        average_completeness = total_completeness / len(numeric_columns)
        return round(average_completeness * 100, 1)
    
    def _calculate_aggregated_data_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall data quality score for aggregated data (0-100)"""
        if not data:
            return 0.0
        
        total_score = 0.0
        operation_count = 0
        
        for operation, operation_data in data.items():
            if 'data' in operation_data and operation_data['data']:
                parsed_data = operation_data['data']
                df = parsed_data['dataframe']
                
                # Calculate score for this operation's data
                score = self._calculate_data_quality_score(df)
                total_score += score
                operation_count += 1
        
        return round(total_score / operation_count, 1) if operation_count > 0 else 0.0
    
    def _get_column_unit(self, column_name: str) -> str:
        """Get unit for a column based on its name"""
        unit_mappings = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'oxygen': 'mg/L',
            'ph': 'pH units',
            'conductivity': 'S/m',
            'depth': 'm',
            'chlorophyll': 'mg/m³',
            'turbidity': 'NTU'
        }
        
        column_lower = column_name.lower()
        for param, unit in unit_mappings.items():
            if param in column_lower:
                return unit
        
        return 'units'
    
    def _format_statistical_results(self, analysis_result: Dict[str, Any], 
                                   stats_params: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format statistical results for user presentation"""
        formatted = {
            'query': query,
            'summary': self._generate_statistical_summary(analysis_result, stats_params),
            'detailed_results': analysis_result,
            'visualizations': []
        }
        
        # Generate visualizations if plotting is available
        if HAS_PLOTTING:
            try:
                visualizations = self._generate_statistical_plots(analysis_result, stats_params)
                formatted['visualizations'] = visualizations
            except Exception as e:
                logger.warning(f"Could not generate visualizations: {e}")
        
        return formatted
    
    def _generate_statistical_summary(self, analysis_result: Dict[str, Any], 
                                    stats_params: Dict[str, Any]) -> str:
        """Generate a natural language summary focusing on temperature data"""
        operations = stats_params['statistical_parameters']['operations']
        statistics = analysis_result.get('statistics', {})
        original_query = stats_params['statistical_parameters'].get('original_query', '').lower()
        
        # Determine which parameter the user is asking about
        target_param = None
        if 'temperature' in original_query:
            target_param = 'temperature'
        elif 'salinity' in original_query:
            target_param = 'salinity'
        elif 'pressure' in original_query:
            target_param = 'pressure'
        elif 'depth' in original_query:
            target_param = 'depth'
        
        summary_parts = []
        
        for operation in operations:
            if operation in statistics and isinstance(statistics[operation], dict) and 'error' not in statistics[operation]:
                op_data = statistics[operation]
                
                # Find the temperature column (or other target parameter)
                temp_column = None
                
                # Look for specific temperature columns based on operation
                for param in op_data.keys():
                    param_lower = param.lower()
                    
                    if operation in ['max', 'maximum']:
                        # Look for "Temperature Max Value" column
                        if 'temperature' in param_lower and 'max' in param_lower and 'value' in param_lower:
                            temp_column = param
                            break
                    elif operation in ['min', 'minimum']:
                        # Look for "Temperature Min Value" column  
                        if 'temperature' in param_lower and 'min' in param_lower and 'value' in param_lower:
                            temp_column = param
                            break
                    elif operation in ['avg', 'average', 'mean']:
                        # For average, look for temperature column (not min/max)
                        if 'temperature' in param_lower and 'max' not in param_lower and 'min' not in param_lower and 'count' not in param_lower:
                            temp_column = param
                            break
                    
                    # Fallback: any temperature column for other target parameters
                    if target_param and target_param in param_lower:
                        temp_column = param
                        break
                
                if temp_column and temp_column in op_data:
                    values = op_data[temp_column]
                    if values.get('value') is not None:
                        # Get depth information from metadata
                        metadata = values.get('metadata', {})
                        depth = metadata.get('depth')
                        depth_str = f" at {depth}m depth" if depth is not None else ""
                        
                        if operation in ['min', 'minimum']:
                            summary_parts.append(
                                f"The minimum temperature was {values['value']:.2f}{values['unit']}{depth_str}"
                            )
                        elif operation in ['max', 'maximum']:
                            summary_parts.append(
                                f"The maximum temperature was {values['value']:.2f}{values['unit']}{depth_str}"
                            )
                        elif operation in ['avg', 'average', 'mean']:
                            summary_parts.append(
                                f"The average temperature was {values['value']:.2f}{values['unit']}{depth_str}"
                            )
        
        return ". ".join(summary_parts) + "." if summary_parts else "Temperature data not found in results."
    
    def _generate_statistical_plots(self, analysis_result: Dict[str, Any], 
                                   stats_params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate statistical plots and return file paths"""
        plots = []
        
        try:
            query = stats_params.get('original_query', '')
            
            # Generate statistical summary plot
            if 'statistics' in analysis_result:
                summary_plot = self.visualizer.create_statistical_summary_plot(
                    analysis_result, query
                )
                if summary_plot:
                    plots.append({
                        'type': 'statistical_summary',
                        'path': summary_plot,
                        'description': 'Statistical summary showing min, max, average, and distribution metrics'
                    })
            
            # Generate time series plot if time series data is available
            if 'time_series_analysis' in analysis_result:
                time_series_data = analysis_result['time_series_analysis']
                if time_series_data and 'aggregated_data' in time_series_data:
                    # Try to plot the first available parameter
                    aggregated_data = time_series_data['aggregated_data']
                    for parameter in aggregated_data.keys():
                        time_series_plot = self.visualizer.create_time_series_plot(
                            time_series_data, parameter
                        )
                        if time_series_plot:
                            plots.append({
                                'type': 'time_series',
                                'parameter': parameter,
                                'path': time_series_plot,
                                'description': f'Time series plot showing {parameter} trends over time'
                            })
                        break  # Only plot the first parameter to avoid too many plots
            
            # Generate correlation matrix if correlation data is available
            if 'statistics' in analysis_result and 'correlation' in analysis_result['statistics']:
                correlation_data = analysis_result['statistics']['correlation']
                if not isinstance(correlation_data, dict) or 'error' not in correlation_data:
                    correlation_plot = self.visualizer.create_correlation_matrix_plot(correlation_data)
                    if correlation_plot:
                        plots.append({
                            'type': 'correlation_matrix',
                            'path': correlation_plot,
                            'description': 'Correlation matrix showing relationships between parameters'
                        })
            
            logger.info(f"Generated {len(plots)} statistical plots")
            
        except Exception as e:
            logger.error(f"Error generating statistical plots: {e}")
        
        return plots
    
    def get_available_statistical_operations(self) -> List[str]:
        """Get list of available statistical operations"""
        return list(self.statistical_operations.keys())
    
    def get_supported_time_windows(self) -> Dict[str, str]:
        """Get supported time aggregation windows"""
        return self.time_windows.copy()