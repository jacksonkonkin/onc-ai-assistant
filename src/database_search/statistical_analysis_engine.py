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
    
    def __init__(self, onc_token: str = None, data_downloader: AdvancedDataDownloader = None):
        """
        Initialize the statistical analysis engine
        
        Args:
            onc_token: ONC API token
            data_downloader: Existing data downloader instance (optional)
        """
        self.data_downloader = data_downloader or AdvancedDataDownloader(onc_token)
        self.parameter_extractor = EnhancedParameterExtractor()
        self.visualizer = StatisticalVisualizer()
        
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
            raw_data_result = self._get_raw_data_for_analysis(data_params)
            
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
            # Read the file and find where data starts (after ## END HEADER)
            with open(csv_file_path, 'r') as f:
                lines = f.readlines()
            
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
            
            # Find the header line (contains column names in quotes)
            header_line_idx = None
            for i in range(len(lines)):
                if lines[i].strip().startswith('#"Time UTC'):
                    header_line_idx = i
                    break
            
            if header_line_idx is not None:
                # Extract column names from the header line
                header_line = lines[header_line_idx].strip()
                # Remove the # and parse the quoted column names
                header_line = header_line[1:]  # Remove #
                column_names = [col.strip().strip('"') for col in header_line.split('", "')]
                # Fix the first and last column names
                if column_names:
                    column_names[0] = column_names[0].strip('"')
                    column_names[-1] = column_names[-1].strip('"')
            else:
                column_names = None
            
            # Read the CSV data starting from the data section
            data_lines = lines[data_start_idx:]
            
            # Create a temporary string buffer with just the data
            from io import StringIO
            data_buffer = StringIO(''.join(data_lines))
            
            # Read into pandas DataFrame with proper column names
            df = pd.read_csv(data_buffer, sep=',', header=None, names=column_names)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            return {
                'dataframe': df,
                'metadata': header_info,
                'total_records': len(df),
                'columns': list(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Error parsing ONC CSV file {csv_file_path}: {e}")
            return None
    
    def _get_raw_data_for_analysis(self, data_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistical data using ONC API's native aggregation capabilities
        
        Args:
            data_params: Data parameters extracted from query
            
        Returns:
            Statistical data result dictionary
        """
        try:
            # Use ONC API's native statistical aggregation
            stats_operations = data_params.get('statistical_operations', ['average'])
            results = {}
            
            # Map our operations to ONC resampling options
            onc_resample_map = {
                'average': 'average',
                'mean': 'average', 
                'avg': 'average',
                'min': 'minMax',  # minMax gives both min and max
                'minimum': 'minMax',
                'max': 'minMax',  # minMax gives both min and max
                'maximum': 'minMax'
            }
            
            # Get data for each requested statistical operation
            for operation in stats_operations:
                if operation in onc_resample_map:
                    resample_type = onc_resample_map[operation]
                    
                    download_params = {
                        'location_code': data_params['location_code'],
                        'device_category': data_params.get('device_category', 'CTD'),
                        'date_from': data_params['start_time'],
                        'date_to': data_params['end_time'],
                        'output_dir': 'output',
                        'quality_control': True,
                        'resample': resample_type  # Use ONC's native statistical aggregation
                    }
                    
                    # Download the aggregated data
                    download_result = self.data_downloader.download_csv_data(**download_params)
                    
                    if download_result['status'] == 'success':
                        csv_files = download_result.get('csv_files', [])
                        if csv_files:
                            # Process the aggregated CSV to extract statistical values
                            for csv_file in csv_files:
                                try:
                                    # Read CSV with header processing for ONC format
                                    aggregated_data = self._parse_onc_csv_file(csv_file)
                                    if aggregated_data:
                                        results[operation] = {
                                            'data': aggregated_data,
                                            'csv_file': csv_file,
                                            'operation': operation,
                                            'resample_type': resample_type
                                        }
                                except Exception as e:
                                    logger.warning(f"Could not process aggregated CSV file {csv_file}: {e}")
            
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
            if col in data.columns:
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
            if col in data.columns:
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
            if col in data.columns:
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