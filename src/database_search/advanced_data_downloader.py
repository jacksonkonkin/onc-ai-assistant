"""
Advanced ONC Data Downloader - Integration of Sprint 3 Scripts
Provides comprehensive data product discovery and download capabilities with query routing
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import json

from .onc_api_client import ONCAPIClient

logger = logging.getLogger(__name__)

try:
    from onc import ONC
    HAS_ONC_PACKAGE = True
except ImportError:
    logger.warning("ONC Python package not available. Using custom API client for basic functionality.")
    HAS_ONC_PACKAGE = False

# Query routing is handled by the main system query router


class AdvancedDataDownloader:
    """
    Advanced data downloader integrating Sprint 3 capabilities
    Supports data product discovery, archive downloads, and CSV exports
    """
    
    def __init__(self, onc_token: str = None):
        """
        Initialize the advanced data downloader
        
        Args:
            onc_token: ONC API token (optional, will use default if not provided)
        """
        # Use token from environment or parameter
        token = onc_token or os.getenv('ONC_API_TOKEN', '45b4e105-43ed-411e-bd1b-1d2799eda3c4')
        
        try:
            if HAS_ONC_PACKAGE:
                # Use official ONC package if available
                self.onc_client = ONC(token)
                self.api_client = None
                logger.info("Advanced Data Downloader initialized with ONC package")
            else:
                # Use custom API client as fallback
                self.onc_client = None
                self.api_client = ONCAPIClient(token)
                logger.info("Advanced Data Downloader initialized with custom API client")
        except Exception as e:
            logger.error(f"Failed to initialize data downloader: {e}")
            raise
    
    # Query classification is handled by the main system query router
    # This class focuses on data download functionality only
    
    def discover_data_products(self, query_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Discover available data products with advanced filtering
        Based on Sprint 3 discover_data_products.py
        
        Args:
            query_params: Dictionary with filtering parameters:
                - locationCode: e.g., "BACAX", "SEVIP"
                - deviceCategoryCode: e.g., "CTD", "ADCP2MHZ", "HYDROPHONE"
                - deviceCode: Specific device code
                - dataProductCode: e.g., "TSSD", "TSSP"
                - extension: e.g., "csv", "mat", "png", "pdf"
                - dataProductName: Search term for product name
                
        Returns:
            Dictionary with discovered products and metadata
        """
        try:
            logger.info(f"Discovering data products with params: {query_params}")
            
            if self.onc_client:
                # Use official ONC package
                params = query_params or {}
                data_products = self.onc_client.getDataProducts(params)
                
                # Organize results by category
                result = {
                    'status': 'success',
                    'total_products': len(data_products),
                    'products': data_products,
                    'categories': self._categorize_data_products(data_products),
                    'filters_applied': params,
                    'discovery_time': datetime.utcnow().isoformat()
                }
                
                logger.info(f"Discovered {len(data_products)} data products")
                return result
            else:
                # Fallback using custom API client
                return self._discover_products_with_custom_client(query_params)
            
        except Exception as e:
            logger.error(f"Error discovering data products: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'products': [],
                'total_products': 0
            }
    
    def _discover_products_with_custom_client(self, query_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Discover data products using custom API client"""
        try:
            params = query_params or {}
            location_code = params.get('locationCode')
            device_category = params.get('deviceCategoryCode')
            
            if not location_code:
                location_code = 'CBYIP'  # Default to Cambridge Bay
            
            # Get available devices as a proxy for data products
            devices = self.api_client.get_devices(location_code, device_category)
            
            # Simulate data product discovery
            simulated_products = []
            for device in devices:
                device_code = device.get('deviceCode', '')
                device_name = device.get('deviceName', 'Unknown Device')
                
                # Create simulated data products for common formats
                for ext in ['csv', 'json']:
                    product = {
                        'dataProductCode': 'TSSD' if ext == 'csv' else 'RAW',
                        'dataProductName': f'Time Series Data - {device_name}',
                        'extension': ext,
                        'hasDeviceData': True,
                        'hasPropertyData': True,
                        'deviceCode': device_code,
                        'locationCode': location_code
                    }
                    simulated_products.append(product)
            
            result = {
                'status': 'success',
                'total_products': len(simulated_products),
                'products': simulated_products,
                'categories': self._categorize_data_products(simulated_products),
                'filters_applied': params,
                'discovery_time': datetime.utcnow().isoformat(),
                'note': 'Simulated data products based on available devices (custom API client)'
            }
            
            logger.info(f"Discovered {len(simulated_products)} simulated data products")
            return result
            
        except Exception as e:
            logger.error(f"Error in custom data products discovery: {e}")
            return {
                'status': 'error',
                'message': f"Custom discovery failed: {str(e)}",
                'products': [],
                'total_products': 0
            }
    
    def get_archived_files(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get list of archived files with advanced filtering
        Based on Sprint 3 download_archived_files.py
        
        Args:
            query_params: Dictionary with parameters:
                - deviceCode: Specific device code
                - deviceCategoryCode: Device category
                - locationCode: Location code
                - extension: File extension filter
                - dateFrom: Start date (ISO format)
                - dateTo: End date (ISO format)
                
        Returns:
            Dictionary with file information
        """
        try:
            logger.info(f"Getting archived files with params: {query_params}")
            
            # Get archived files
            archived_files = self.onc_client.getArchivefile(query_params)
            
            # Process and organize results
            result = {
                'status': 'success',
                'total_files': len(archived_files),
                'files': archived_files,
                'size_summary': self._calculate_size_summary(archived_files),
                'time_range': self._extract_time_range(archived_files),
                'file_types': self._get_file_type_summary(archived_files),
                'query_params': query_params
            }
            
            logger.info(f"Found {len(archived_files)} archived files")
            return result
            
        except Exception as e:
            logger.error(f"Error getting archived files: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'files': [],
                'total_files': 0
            }
    
    def download_data_product(self, product_params: Dict[str, Any], 
                             output_dir: str = "downloads") -> Dict[str, Any]:
        """
        Download data products with processing options
        Based on Sprint 3 download_data_products.py
        
        Args:
            product_params: Dictionary with parameters:
                - locationCode: Location code
                - deviceCategoryCode: Device category
                - dataProductCode: Product code (e.g., "TSSD", "TSSP")
                - extension: File format
                - dateFrom: Start date
                - dateTo: End date
                - dpo_qualityControl: Quality control option
                - dpo_resample: Resampling option
                - dpo_dataGaps: Data gaps option
            output_dir: Directory to save downloaded files
            
        Returns:
            Dictionary with download results
        """
        try:
            logger.info(f"Downloading data product with params: {product_params}")
            
            # Ensure output directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            if self.onc_client:
                # Use official ONC package
                result = self.onc_client.orderDataProduct(
                    product_params, 
                    includeMetadataFile=True
                )
                
                # Process download result
                download_info = {
                    'status': 'success',
                    'download_result': result,
                    'output_directory': output_dir,
                    'product_params': product_params,
                    'download_time': datetime.utcnow().isoformat()
                }
                
                # Add file information if available
                if isinstance(result, dict) and 'files' in result:
                    download_info['downloaded_files'] = result['files']
                    download_info['file_count'] = len(result['files'])
                elif isinstance(result, list):
                    download_info['downloaded_files'] = result
                    download_info['file_count'] = len(result)
                
                logger.info(f"Successfully downloaded data product")
                return download_info
            else:
                # Fallback using custom API client
                return self._download_data_product_with_custom_client(product_params, output_dir)
            
        except Exception as e:
            logger.error(f"Error downloading data product: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'download_result': None
            }
    
    def _download_data_product_with_custom_client(self, product_params: Dict[str, Any], 
                                                 output_dir: str) -> Dict[str, Any]:
        """Download data product using custom API client as fallback"""
        try:
            location_code = product_params.get('locationCode')
            device_category = product_params.get('deviceCategoryCode')
            date_from = product_params.get('dateFrom')
            date_to = product_params.get('dateTo')
            extension = product_params.get('extension', 'csv')
            
            if extension == 'csv':
                # For CSV, use our CSV download method
                return self._download_csv_with_custom_client(
                    location_code, device_category, date_from, date_to,
                    output_dir, True, "none"
                )
            else:
                # For other formats, return a message that we can't handle them
                return {
                    'status': 'error',
                    'message': f"Custom API client only supports CSV downloads. Requested format: {extension}",
                    'download_result': None
                }
                
        except Exception as e:
            logger.error(f"Error in custom data product download: {e}")
            return {
                'status': 'error',
                'message': f"Custom data product download failed: {str(e)}",
                'download_result': None
            }
    
    def bulk_download_archived_files(self, query_params: Dict[str, Any], 
                                   output_dir: str = "downloads", 
                                   generate_urls_only: bool = False) -> Dict[str, Any]:
        """
        Bulk download archived files or generate download URLs
        
        Args:
            query_params: Query parameters for file selection
            output_dir: Directory to save files
            generate_urls_only: If True, only generate URLs without downloading
            
        Returns:
            Dictionary with download results or URLs
        """
        try:
            if generate_urls_only:
                # Generate download URLs for external download manager
                urls = self.onc_client.getArchivefileUrls(query_params, joinedWithNewline=True)
                
                return {
                    'status': 'success',
                    'download_urls': urls.split('\n') if isinstance(urls, str) else urls,
                    'url_count': len(urls.split('\n')) if isinstance(urls, str) else len(urls),
                    'query_params': query_params
                }
            else:
                # Direct bulk download
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                # Download all matching files
                download_result = self.onc_client.downloadDirectArchivefile(query_params)
                
                return {
                    'status': 'success',
                    'download_result': download_result,
                    'output_directory': output_dir,
                    'query_params': query_params,
                    'download_time': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in bulk download: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'download_result': None
            }
    
    def download_csv_data(self, location_code: str, device_category: str,
                         date_from: str, date_to: str,
                         output_dir: str = "csv_downloads",
                         quality_control: bool = True,
                         resample: str = "none") -> Dict[str, Any]:
        """
        Download CSV data for a specific location and device
        Optimized for CSV data export functionality
        
        Args:
            location_code: Location code (e.g., "SEVIP", "BACAX")
            device_category: Device category (e.g., "CTD", "ADCP2MHZ")
            date_from: Start date in ISO format
            date_to: End date in ISO format
            output_dir: Directory to save CSV files
            quality_control: Apply quality control
            resample: Resampling option ("none", "average", "minMax", "minMaxAvg")
            
        Returns:
            Dictionary with download results and CSV file paths
        """
        try:
            logger.info(f"Downloading CSV data for {device_category} at {location_code}")
            
            if self.onc_client:
                # Use official ONC package
                return self._download_csv_with_onc_package(
                    location_code, device_category, date_from, date_to, 
                    output_dir, quality_control, resample
                )
            else:
                # Use custom API client fallback
                return self._download_csv_with_custom_client(
                    location_code, device_category, date_from, date_to,
                    output_dir, quality_control, resample
                )
            
        except Exception as e:
            logger.error(f"Error downloading CSV data: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'csv_files': []
            }
    
    def _download_csv_with_onc_package(self, location_code: str, device_category: str,
                                      date_from: str, date_to: str, output_dir: str,
                                      quality_control: bool, resample: str) -> Dict[str, Any]:
        """Download CSV using official ONC package"""
        # Parameters for CSV data download
        params = {
            "locationCode": location_code,
            "deviceCategoryCode": device_category,
            "dataProductCode": "TSSD",  # Time Series Scalar Data
            "extension": "csv",
            "dateFrom": date_from,
            "dateTo": date_to,
            "dpo_qualityControl": 1 if quality_control else 0,
            "dpo_resample": resample,
            "dpo_dataGaps": 0
        }
        
        # Download the CSV data
        download_result = self.download_data_product(params, output_dir)
        
        if download_result['status'] == 'success':
            # Process CSV files for additional metadata
            csv_files = self._find_csv_files(download_result, output_dir)
            csv_info = self._analyze_csv_files(csv_files)
            
            download_result.update({
                'csv_files': csv_files,
                'csv_analysis': csv_info,
                'location_code': location_code,
                'device_category': device_category,
                'date_range': f"{date_from} to {date_to}"
            })
        
        return download_result
    
    def _download_csv_with_custom_client(self, location_code: str, device_category: str,
                                        date_from: str, date_to: str, output_dir: str,
                                        quality_control: bool, resample: str) -> Dict[str, Any]:
        """Download CSV using custom API client as fallback"""
        try:
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Use our existing API client to get scalar data
            logger.info("Using custom API client for CSV data download")
            
            # Get devices at location
            devices = self.api_client.get_devices(location_code, device_category)
            if not devices:
                return {
                    'status': 'error',
                    'message': f"No {device_category} devices found at {location_code}",
                    'csv_files': []
                }
            
            csv_files = []
            all_data = []
            
            # Get data from each device
            for device in devices[:3]:  # Limit to first 3 devices for demo
                device_code = device.get('deviceCode')
                if not device_code:
                    continue
                
                try:
                    # Get scalar data for temperature property
                    property_code = 'seawatertemperature' if 'temperature' in device_category.lower() else None
                    
                    data = self.api_client.get_scalar_data(
                        device_code=device_code,
                        property_code=property_code,
                        date_from=date_from,
                        date_to=date_to,
                        row_limit=10000
                    )
                    
                    if data and 'sensorData' in data:
                        sensor_data = data['sensorData']
                        if sensor_data:
                            # Convert to CSV format
                            csv_filename = f"{device_code}_{location_code}_{date_from[:10]}_to_{date_to[:10]}.csv"
                            csv_path = os.path.join(output_dir, csv_filename)
                            
                            # Create DataFrame and save as CSV
                            df = pd.DataFrame(sensor_data)
                            df.to_csv(csv_path, index=False)
                            
                            csv_files.append(csv_path)
                            all_data.extend(sensor_data)
                            
                            logger.info(f"Created CSV file: {csv_filename} with {len(sensor_data)} records")
                    
                except Exception as e:
                    logger.warning(f"Could not get data for device {device_code}: {e}")
                    continue
            
            if csv_files:
                return {
                    'status': 'success',
                    'message': f"Downloaded CSV data to {len(csv_files)} files",
                    'csv_files': csv_files,
                    'output_directory': output_dir,
                    'file_count': len(csv_files),
                    'location_code': location_code,
                    'device_category': device_category,
                    'date_range': f"{date_from} to {date_to}",
                    'total_records': len(all_data)
                }
            else:
                return {
                    'status': 'error',
                    'message': f"No data available for {device_category} at {location_code} for the specified date range",
                    'csv_files': []
                }
                
        except Exception as e:
            logger.error(f"Error in custom CSV download: {e}")
            return {
                'status': 'error',
                'message': f"CSV download failed: {str(e)}",
                'csv_files': []
            }
    
    def get_data_summary(self, location_code: str = None, 
                        device_category: str = None,
                        date_from: str = None, date_to: str = None) -> Dict[str, Any]:
        """
        Get comprehensive data availability summary
        
        Args:
            location_code: Optional location filter
            device_category: Optional device category filter  
            date_from: Optional start date filter
            date_to: Optional end date filter
            
        Returns:
            Summary of available data and products
        """
        try:
            # Build filter parameters
            filters = {}
            if location_code:
                filters['locationCode'] = location_code
            if device_category:
                filters['deviceCategoryCode'] = device_category
            
            # Get data products
            products_result = self.discover_data_products(filters)
            
            # Get archived files if date range provided
            archived_files = {}
            if date_from and date_to:
                archive_filters = filters.copy()
                archive_filters.update({
                    'dateFrom': date_from,
                    'dateTo': date_to
                })
                archived_files = self.get_archived_files(archive_filters)
            
            # Compile summary
            summary = {
                'status': 'success',
                'location_code': location_code,
                'device_category': device_category,
                'date_range': f"{date_from} to {date_to}" if date_from and date_to else "All time",
                'data_products': {
                    'total': products_result.get('total_products', 0),
                    'categories': products_result.get('categories', {}),
                    'available_formats': self._extract_available_formats(products_result.get('products', []))
                },
                'archived_files': {
                    'total': archived_files.get('total_files', 0),
                    'size_summary': archived_files.get('size_summary', {}),
                    'file_types': archived_files.get('file_types', {})
                },
                'summary_time': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    # Helper methods
    def _categorize_data_products(self, products: List[Dict]) -> Dict[str, int]:
        """Categorize data products by type"""
        categories = {}
        for product in products:
            category = product.get('dataProductCode', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _calculate_size_summary(self, files: List[Dict]) -> Dict[str, Any]:
        """Calculate size summary for archived files"""
        if not files:
            return {'total_size': 0, 'average_size': 0, 'file_count': 0}
        
        total_size = sum(file.get('fileSize', 0) for file in files)
        return {
            'total_size': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'average_size': total_size // len(files) if files else 0,
            'file_count': len(files)
        }
    
    def _extract_time_range(self, files: List[Dict]) -> Dict[str, str]:
        """Extract time range from archived files"""
        if not files:
            return {'earliest': None, 'latest': None}
        
        timestamps = []
        for file in files:
            if 'dateFrom' in file:
                timestamps.append(file['dateFrom'])
            if 'dateTo' in file:
                timestamps.append(file['dateTo'])
        
        if timestamps:
            return {
                'earliest': min(timestamps),
                'latest': max(timestamps)
            }
        return {'earliest': None, 'latest': None}
    
    def _get_file_type_summary(self, files: List[Dict]) -> Dict[str, int]:
        """Get summary of file types"""
        types = {}
        for file in files:
            ext = file.get('extension', 'unknown')
            types[ext] = types.get(ext, 0) + 1
        return types
    
    def _find_csv_files(self, download_result: Dict, output_dir: str) -> List[str]:
        """Find CSV files from download result"""
        csv_files = []
        
        # Check download result for file paths
        if 'downloaded_files' in download_result:
            for file_info in download_result['downloaded_files']:
                if isinstance(file_info, dict):
                    filename = file_info.get('filename', '')
                elif isinstance(file_info, str):
                    filename = file_info
                else:
                    continue
                
                if filename.endswith('.csv'):
                    csv_files.append(os.path.join(output_dir, filename))
        
        # Also scan output directory
        output_path = Path(output_dir)
        if output_path.exists():
            csv_files.extend([str(f) for f in output_path.glob("*.csv")])
        
        return list(set(csv_files))  # Remove duplicates
    
    def _analyze_csv_files(self, csv_files: List[str]) -> Dict[str, Any]:
        """Analyze CSV files for metadata"""
        analysis = {
            'file_count': len(csv_files),
            'files': []
        }
        
        for csv_file in csv_files:
            try:
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    file_info = {
                        'filename': os.path.basename(csv_file),
                        'path': csv_file,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': df.columns.tolist(),
                        'file_size_bytes': os.path.getsize(csv_file)
                    }
                    analysis['files'].append(file_info)
            except Exception as e:
                logger.warning(f"Could not analyze CSV file {csv_file}: {e}")
        
        return analysis
    
    def _extract_available_formats(self, products: List[Dict]) -> List[str]:
        """Extract available file formats from products"""
        formats = set()
        for product in products:
            ext = product.get('extension')
            if ext:
                formats.add(ext)
        return sorted(list(formats))