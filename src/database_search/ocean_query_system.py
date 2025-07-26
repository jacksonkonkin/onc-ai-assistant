#!/usr/bin/env python3
"""
Ocean Query System - Complete Pipeline
Natural Language Query → Parameter Extraction → ONC API Call → Raw JSON Response
"""

import json
import sys
import time
import logging
from datetime import timedelta
from typing import Dict, Any, Optional, List

from .enhanced_parameter_extractor import EnhancedParameterExtractor
from .onc_api_client import ONCAPIClient
from .enhanced_response_formatter import EnhancedResponseFormatter
from .advanced_data_downloader import AdvancedDataDownloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OceanQuerySystem:
    """Complete ocean data query system"""
    
    def __init__(self, onc_token: str = None, llm_wrapper=None):
        """
        Initialize the complete query system
        
        Args:
            onc_token: ONC API token (optional, will use default if not provided)
            llm_wrapper: LLM wrapper for enhanced response formatting
        """
        try:
            self.api_client = ONCAPIClient(onc_token)
            self.extractor = EnhancedParameterExtractor(onc_client=self.api_client)
            
            # Initialize enhanced response formatter if LLM wrapper is available
            self.enhanced_formatter = None
            if llm_wrapper:
                self.enhanced_formatter = EnhancedResponseFormatter(llm_wrapper)
                logger.info("Enhanced response formatting enabled")
            
            # Initialize advanced data downloader
            self.data_downloader = AdvancedDataDownloader(onc_token)
            logger.info("Advanced data downloader enabled")
            
            logger.info("Ocean Query System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    def process_query(self, query: str, include_metadata: bool = True, 
                     query_type: str = "data", row_limit: int = 1000, 
                     max_devices: int = None, parallel: bool = False,
                     use_pagination: bool = False, page_size: int = 500, 
                     max_pages: int = 10) -> Dict[str, Any]:
        """
        Process a natural language query and return ONC API data
        
        Args:
            query: Natural language query
            include_metadata: Whether to include processing metadata
            query_type: Type of query - "data" for sensor data, "device_discovery" for device listing
            row_limit: Maximum rows per device (default: 1000)
            max_devices: Maximum number of devices to query (None = all)
            parallel: Whether to query devices in parallel
            use_pagination: Whether to use time-based pagination
            page_size: Rows per time chunk when using pagination
            max_pages: Maximum number of time chunks
            
        Returns:
            Complete response with data and metadata
        """
        start_time = time.time()
        logger.info(f"Processing {query_type} query: '{query}'")
        
        # Route to appropriate processing method
        if query_type == "device_discovery":
            return self.process_device_discovery_query(query, include_metadata)
        elif query_type == "data_products":
            return self.process_data_products_query(query, include_metadata)
        elif query_type == "data_download":
            return self.process_data_download_query(query, include_metadata, row_limit)
        elif query_type == "data_preview":
            return self.process_data_preview_query(query, include_metadata, row_limit)
        else:
            return self.process_data_query(query, include_metadata, row_limit, max_devices, 
                                         parallel, use_pagination, page_size, max_pages)

    def process_device_discovery_query(self, query: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Process a device discovery query to find available devices/sensors
        
        Args:
            query: Natural language query about devices
            include_metadata: Whether to include processing metadata
            
        Returns:
            Response with device information
        """
        start_time = time.time()
        logger.info(f"Processing device discovery query: '{query}'")
        
        # Step 1: Extract location and device type from query
        logger.info("Step 1: Extracting parameters for device discovery...")
        extraction_result = self.extractor.extract_parameters(query)
        
        if extraction_result["status"] != "success":
            return {
                "status": "error",
                "stage": "parameter_extraction", 
                "message": extraction_result.get("message", "Parameter extraction failed"),
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "device_discovery",
                    "extraction_result": extraction_result,
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        params = extraction_result["parameters"]
        logger.info(f"Extracted parameters for device discovery: {params}")
        
        # Step 2: Discover devices using ONC API
        logger.info("Step 2: Discovering devices...")
        
        try:
            # Use device discovery methods instead of data search
            location_code = params["location_code"]
            device_category = params.get("device_category")
            property_code = params.get("property_code")
            
            # For device discovery queries, don't filter by property unless specifically requested
            # If the user asks "what CTD devices are available", they want all CTD devices,
            # not CTD devices that measure a specific property
            device_discovery_keywords = ['devices', 'sensors', 'instruments', 'available', 'deployed', 'what']
            original_query = params.get("original_query", "").lower() if "original_query" in params else query.lower()
            
            # If this is clearly a device discovery query, ignore property filtering
            is_device_discovery = any(keyword in original_query for keyword in device_discovery_keywords)
            use_property_filter = property_code if not is_device_discovery else None
            
            logger.info(f"Device discovery mode: {is_device_discovery}, using property filter: {use_property_filter}")
            
            # Search devices with optional property code filtering
            devices = self.api_client.find_cambridge_bay_devices(
                device_category=device_category,
                property_code=use_property_filter
            )
            
            # Filter by location if not all Cambridge Bay
            if location_code != "CBYIP":
                devices = [d for d in devices if d.get('_location_info', {}).get('locationCode') == location_code]
            
            total_time = time.time() - start_time
            
            if devices:
                return {
                    "status": "success",
                    "message": f"Found {len(devices)} devices",
                    "data": devices,
                    "metadata": {
                        "query": query,
                        "query_type": "device_discovery",
                        "location_code": location_code,
                        "device_category": device_category,
                        "property_code": property_code,
                        "devices_found": len(devices),
                        "execution_time": round(total_time, 2)
                    } if include_metadata else None
                }
            else:
                return {
                    "status": "no_devices",
                    "message": f"No devices found for the specified criteria",
                    "data": [],
                    "metadata": {
                        "query": query,
                        "query_type": "device_discovery", 
                        "location_code": location_code,
                        "device_category": device_category,
                        "property_code": property_code,
                        "execution_time": round(total_time, 2)
                    } if include_metadata else None
                }
                
        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return {
                "status": "error",
                "stage": "device_discovery",
                "message": f"Device discovery failed: {str(e)}",
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "device_discovery",
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }

    def process_data_products_query(self, query: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Process a data products discovery or download query
        
        Args:
            query: Natural language query
            include_metadata: Whether to include processing metadata
            
        Returns:
            Dictionary with data products information or download status
        """
        start_time = time.time()
        logger.info(f"Processing data products query: {query}")
        
        # Step 1: Extract parameters from query
        logger.info("Step 1: Extracting parameters...")
        extraction_result = self.extractor.extract_parameters(query)
        
        if extraction_result["status"] != "success":
            return {
                "status": "error", 
                "stage": "parameter_extraction",
                "message": extraction_result["message"],
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "data_products",
                    "extraction_result": extraction_result,
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        params = extraction_result["parameters"]
        logger.info(f"Extracted parameters for data products: {params}")
        
        # Step 2: Determine if this is discovery or download request
        logger.info("Step 2: Processing data products request...")
        
        try:
            location_code = params["location_code"]
            device_category = params.get("device_category")
            property_code = params.get("property_code")
            temporal_reference = params.get("temporal_reference")
            
            # Check if this is a download request or discovery based on query intent and temporal reference
            download_keywords = ['download', 'export', 'get data', 'retrieve data', 'order']
            discovery_keywords = ['what', 'available', 'show me', 'list', 'find', 'discover']
            
            has_download_intent = any(keyword in query.lower() for keyword in download_keywords)
            has_discovery_intent = any(keyword in query.lower() for keyword in discovery_keywords)
            has_specific_date = temporal_reference and temporal_reference != "latest" and len(temporal_reference) >= 10
            
            # Download request if: explicit download keywords OR specific date mentioned
            is_download_request = has_download_intent or (has_specific_date and not has_discovery_intent)
            
            if is_download_request:
                # This is a data product download request
                return self._process_data_product_download(
                    query, params, include_metadata, start_time
                )
            else:
                # This is a data products discovery request
                return self._process_data_products_discovery(
                    query, params, include_metadata, start_time
                )
                
        except Exception as e:
            logger.error(f"Data products processing failed: {e}")
            return {
                "status": "error",
                "stage": "data_products_processing",
                "message": f"Data products processing failed: {str(e)}",
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "data_products",
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
    
    def _process_data_products_discovery(self, query: str, params: Dict[str, Any], 
                                       include_metadata: bool, start_time: float) -> Dict[str, Any]:
        """Process data products discovery request"""
        location_code = params["location_code"]
        device_category = params.get("device_category")
        property_code = params.get("property_code")
        
        # Map property code to common data product codes
        data_product_mapping = {
            'seawatertemperature': 'TSSD',  # Time Series Scalar Data
            'salinity': 'TSSD',
            'pressure': 'TSSD',
            'conductivity': 'TSSD',
            'ph': 'TSSD',
            'oxygen': 'TSSD'
        }
        
        data_product_code = data_product_mapping.get(property_code) if property_code else None
        
        # Discover data products
        if location_code == "CBYIP" or location_code.startswith("CBYSS"):
            # Use Cambridge Bay specific method
            data_products = self.api_client.discover_cambridge_bay_data_products(
                device_category=device_category,
                data_product_code=data_product_code
            )
        else:
            # Use general method
            data_products = self.api_client.get_data_products(
                location_code=location_code,
                device_category=device_category,
                data_product_code=data_product_code
            )
        
        total_time = time.time() - start_time
        
        if data_products:
            return {
                "status": "success",
                "message": f"Found {len(data_products)} available data products",
                "data": data_products,
                "metadata": {
                    "query": query,
                    "query_type": "data_products_discovery",
                    "location_code": location_code,
                    "device_category": device_category,
                    "property_code": property_code,
                    "data_product_code": data_product_code,
                    "products_found": len(data_products),
                    "execution_time": round(total_time, 2)
                } if include_metadata else None
            }
        else:
            return {
                "status": "no_products",
                "message": "No data products found for the specified criteria",
                "data": [],
                "metadata": {
                    "query": query,
                    "query_type": "data_products_discovery",
                    "location_code": location_code,
                    "device_category": device_category,
                    "property_code": property_code,
                    "execution_time": round(total_time, 2)
                } if include_metadata else None
            }
    
    def _process_data_product_download(self, query: str, params: Dict[str, Any], 
                                     include_metadata: bool, start_time: float) -> Dict[str, Any]:
        """Process data product download request"""
        location_code = params["location_code"]
        device_category = params.get("device_category")
        property_code = params.get("property_code")
        temporal_reference = params.get("temporal_reference")
        
        # Map property code to data product code
        data_product_mapping = {
            'seawatertemperature': 'TSSD',
            'salinity': 'TSSD',
            'pressure': 'TSSD',
            'conductivity': 'TSSD',
            'ph': 'TSSD',
            'oxygen': 'TSSD'
        }
        
        data_product_code = data_product_mapping.get(property_code, 'TSSD')
        
        # Parse temporal reference to date range
        # This is a simplified implementation - you may want to enhance this
        from datetime import datetime, timedelta
        
        if temporal_reference and temporal_reference != "latest":
            try:
                # Try to parse as a specific date
                if len(temporal_reference) == 10:  # YYYY-MM-DD format
                    date_from = f"{temporal_reference}T00:00:00.000Z"
                    # Default to next day for single date queries
                    date_obj = datetime.fromisoformat(temporal_reference)
                    next_day = date_obj + timedelta(days=1)
                    date_to = f"{next_day.strftime('%Y-%m-%d')}T00:00:00.000Z"
                else:
                    # Default to last 24 hours if we can't parse
                    end_time = datetime.now()
                    start_time_dt = end_time - timedelta(hours=24)
                    date_from = start_time_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                    date_to = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            except:
                # Fallback to last 24 hours
                end_time = datetime.now()
                start_time_dt = end_time - timedelta(hours=24)
                date_from = start_time_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                date_to = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        else:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time_dt = end_time - timedelta(hours=24)
            date_from = start_time_dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            date_to = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        # Use data downloader to get CSV data
        download_result = self.data_downloader.download_csv_data(
            location_code=location_code,
            device_category=device_category,
            date_from=date_from,
            date_to=date_to,
            output_dir="csv_downloads",
            quality_control=True,
            resample="none"
        )
        
        total_time = time.time() - start_time
        
        if download_result.get('status') == 'success':
            # Extract download information from download_result
            download_data = download_result.get('download_result', {})
            dp_request_id = download_data.get('dpRequestId')
            estimated_size = download_data.get('estimatedFileSize', 'Unknown')
            estimated_time = download_data.get('estimatedProcessingTime', 'Unknown')
            
            # Generate download status URL
            download_status_url = self.api_client.generate_download_status_url(str(dp_request_id)) if dp_request_id else None
            
            # Enhanced result with download information
            enhanced_result = download_data.copy() if download_data else {}
            enhanced_result['download_info'] = {
                'dp_request_id': dp_request_id,
                'estimated_file_size': estimated_size,
                'estimated_processing_time': estimated_time,
                'status_check_url': download_status_url,
                'instructions': 'Data product is being processed. Check the status URL for download links when ready.'
            }
            
            return {
                "status": "success",
                "message": f"Data product download initiated successfully (Request ID: {dp_request_id})",
                "data": enhanced_result,
                "metadata": {
                    "query": query,
                    "query_type": "data_product_download",
                    "location_code": location_code,
                    "device_category": device_category,
                    "property_code": property_code,
                    "data_product_code": data_product_code,
                    "date_from": date_from,
                    "date_to": date_to,
                    "dp_request_id": dp_request_id,
                    "execution_time": round(total_time, 2)
                } if include_metadata else None
            }
        else:
            return {
                "status": "error",
                "message": f"Failed to initiate data product download: {download_result.get('message', 'Unknown error')}",
                "data": download_result,
                "metadata": {
                    "query": query,
                    "query_type": "data_product_download",
                    "execution_time": round(total_time, 2)
                } if include_metadata else None
            }

    def process_data_query(self, query: str, include_metadata: bool = True,
                          row_limit: int = 1000, max_devices: int = None, 
                          parallel: bool = False, use_pagination: bool = False,
                          page_size: int = 500, max_pages: int = 10) -> Dict[str, Any]:
        """
        Process a sensor data query to download actual measurements
        
        Args:
            query: Natural language query
            include_metadata: Whether to include processing metadata
            row_limit: Maximum rows per device (default: 1000)
            max_devices: Maximum number of devices to query (None = all)
            parallel: Whether to query devices in parallel
            use_pagination: Whether to use time-based pagination
            page_size: Rows per time chunk when using pagination
            max_pages: Maximum number of time chunks
            
        Returns:
            Complete response with sensor data and metadata
        """
        start_time = time.time()
        logger.info(f"Processing data query: '{query}'")
        
        # Step 1: Extract parameters from natural language
        logger.info("Step 1: Extracting parameters...")
        extraction_result = self.extractor.extract_parameters(query)
        
        if extraction_result["status"] != "success":
            return {
                "status": "error",
                "stage": "parameter_extraction",
                "message": extraction_result.get("message", "Parameter extraction failed"),
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "data",
                    "extraction_result": extraction_result,
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        params = extraction_result["parameters"]
        logger.info(f"Extracted parameters: {params}")
        
        # Step 2: Call ONC API with extracted parameters
        logger.info("Step 2: Calling ONC API...")
        
        try:
            # Choose the appropriate API method based on parameters
            if use_pagination:
                api_result = self.api_client.get_paginated_data(
                    location_code=params["location_code"],
                    device_category=params["device_category"],
                    property_code=params["property_code"],
                    date_from=params["start_time"],
                    date_to=params["end_time"],
                    page_size=page_size,
                    max_pages=max_pages
                )
            else:
                api_result = self.api_client.search_data_range(
                    location_code=params["location_code"],
                    device_category=params["device_category"],
                    property_code=params["property_code"],
                    date_from=params["start_time"],
                    date_to=params["end_time"],
                    row_limit=row_limit,
                    max_devices=max_devices,
                    parallel=parallel
                )
            
            logger.info(f"API call completed with status: {api_result['status']}")
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {
                "status": "error", 
                "stage": "api_call",
                "message": f"ONC API call failed: {str(e)}",
                "data": None,
                "metadata": {
                    "query": query,
                    "extracted_parameters": params,
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        # Step 3: Format and return complete response
        total_time = time.time() - start_time
        logger.info(f"Query processing completed in {total_time:.2f}s")
        
        # Build final response
        response = {
            "status": api_result["status"],
            "message": api_result["message"],
            "data": api_result["data"]
        }
        
        # Always include raw_api_responses for debugging/educational purposes
        if "raw_api_responses" in api_result:
            response["raw_api_responses"] = api_result["raw_api_responses"]
        
        if include_metadata:
            response["metadata"] = {
                "query": query,
                "extracted_parameters": params,
                "extraction_metadata": extraction_result.get("metadata", {}),
                "api_metadata": api_result.get("metadata", {}),
                "total_execution_time": round(total_time, 2)
            }
        
        return response

    def process_data_preview_query(self, query: str, include_metadata: bool = True, 
                                  row_limit: int = 1000) -> Dict[str, Any]:
        """
        Process a data preview query - show data first, then offer download option
        OR handle actual download if download flag is detected
        
        Args:
            query: Natural language query
            include_metadata: Whether to include processing metadata
            row_limit: Maximum rows for data preview
            
        Returns:
            Response with data preview and download offer OR actual download
        """
        start_time = time.time()
        logger.info(f"Processing data preview query: '{query}'")
        
        # Step 1: Extract parameters to check download flag
        logger.info("Step 1: Extracting parameters...")
        extraction_result = self.extractor.extract_parameters(query)
        
        if extraction_result["status"] != "success":
            return {
                "status": "error",
                "stage": "parameter_extraction",
                "message": extraction_result["message"],
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "data_preview",
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        params = extraction_result["parameters"]
        download_requested = params.get("download_requested", False)
        
        # Step 2: Route based on download flag
        if download_requested:
            logger.info("Download requested - routing to CSV download")
            return self._process_csv_download(query, params, include_metadata, start_time)
        else:
            logger.info("No download requested - showing data preview")
            return self._process_data_preview_only(query, params, include_metadata, start_time, row_limit)
    
    def _process_data_preview_only(self, query: str, params: Dict[str, Any], 
                                  include_metadata: bool, start_time: float, 
                                  row_limit: int) -> Dict[str, Any]:
        """Process data preview without download."""
        
        # Get the data (same as regular data query but limited)
        data_result = self.process_data_query(
            query, 
            include_metadata=False,  # We'll add our own metadata
            row_limit=min(row_limit, 100),  # Limit preview to 100 rows max
            max_devices=3,  # Limit to 3 devices for preview
            parallel=False,
            use_pagination=False
        )
        
        if data_result["status"] != "success":
            # If data query failed, return the failure
            return data_result
        
        # Modify the response to include download offer
        total_time = time.time() - start_time
        
        # Create enhanced response with download offer
        enhanced_response = {
            "status": "success",
            "message": data_result["message"],
            "data": data_result["data"],
            "preview_info": {
                "is_preview": True,
                "preview_rows": len(data_result.get("data", [])),
                "download_available": True,
                "download_offer": "Would you like me to download this data as CSV files for you?"
            }
        }
        
        # Copy over raw API responses for formatting
        if "raw_api_responses" in data_result:
            enhanced_response["raw_api_responses"] = data_result["raw_api_responses"]
        
        if include_metadata:
            enhanced_response["metadata"] = {
                "query": query,
                "query_type": "data_preview",
                "execution_time": round(total_time, 2),
                "original_metadata": data_result.get("metadata", {}),
                "download_requested": False
            }
        
        return enhanced_response

    def get_latest_data(self, query: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get the most recent data for a query
        
        Args:
            query: Natural language query
            hours_back: How many hours back to search
            
        Returns:
            Latest data response
        """
        logger.info(f"Getting latest data for: '{query}' ({hours_back}h back)")
        
        # Extract parameters
        extraction_result = self.extractor.extract_parameters(query)
        if extraction_result["status"] != "success":
            return extraction_result
        
        params = extraction_result["parameters"]
        
        # Get latest data using API client
        try:
            result = self.api_client.get_latest_data(
                location_code=params["location_code"],
                device_category=params["device_category"],
                property_code=params["property_code"],
                hours_back=hours_back
            )
            
            # Add extraction info to metadata
            if "metadata" in result:
                result["metadata"]["extracted_parameters"] = params
                result["metadata"]["query"] = query
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get latest data: {e}")
            return {
                "status": "error",
                "message": f"Failed to get latest data: {str(e)}",
                "data": None
            }

    def format_enhanced_response(self, response: Dict[str, Any], 
                               conversation_context: str = "") -> str:
        """
        Format response using enhanced natural language formatting.
        
        Args:
            response: System response dictionary
            conversation_context: Previous conversation context
            
        Returns:
            Enhanced natural language response
        """
        if self.enhanced_formatter:
            try:
                # Get original query from metadata
                metadata = response.get("metadata", {})
                original_query = metadata.get("query", "")
                
                enhanced_response = self.enhanced_formatter.format_enhanced_response(
                    response, conversation_context, original_query
                )
                
                # If the enhanced response indicates an error, fall back to technical format
                if "unexpected error while formatting" in enhanced_response:
                    logger.warning("Enhanced formatting failed, falling back to technical format")
                    return self.format_response_for_display(response, show_api_calls=True)
                
                return enhanced_response
                
            except Exception as e:
                logger.error(f"Enhanced formatting failed: {e}")
                logger.info("Falling back to technical formatting")
                return self.format_response_for_display(response, show_api_calls=True)
        else:
            # Fallback to regular formatting
            return self.format_response_for_display(response, show_api_calls=True)

    def format_response_for_display(self, response: Dict[str, Any], include_raw_data: bool = False, 
                                   show_api_calls: bool = False) -> str:
        """
        Format response for enhanced human-readable display
        
        Args:
            response: System response dictionary
            include_raw_data: Whether to include full raw API responses
            show_api_calls: Whether to show API calls made
            
        Returns:
            Formatted string for display
        """
        # Handle error cases
        if response["status"] == "error":
            return f"ERROR: {response['message']}"
        
        if response["status"] == "no_data":
            # Enhanced no data response
            meta = response.get("metadata", {})
            extracted_params = meta.get('extracted_parameters', {})
            location = extracted_params.get('location_code', 'Unknown')
            device = extracted_params.get('device_category', 'Unknown')
            property_code = extracted_params.get('property_code', 'Unknown')
            
            lines = [
                f"NO DATA: No {property_code} data found from {device} devices at {location}.",
                "",
                self._format_query_details(response),
                self._format_suggestions(response)
            ]
            return "\n".join(lines)
        
        if response["status"] != "success" or not response["data"]:
            return f"WARNING: {response.get('message', 'Unknown status')}"
        
        # Check if this is a device discovery response before trying to format as sensor data
        metadata = response.get("metadata", {})
        if metadata.get("query_type") == "device_discovery":
            # This is a device discovery response - format it appropriately
            return self._format_device_discovery_response(response)
        
        # Format successful sensor data response (not device discovery)
        formatted_data = self.api_client.format_sensor_data(response["data"])
        
        if not formatted_data:
            return "INFO: No sensor data found in response"
        
        # Start with one-sentence summary
        lines = [self._format_summary_sentence(response, formatted_data)]
        lines.append("")
        
        # Add API calls information only
        lines.append(self._format_api_calls(response))
        
        return "\n".join(lines)

    def _format_summary_sentence(self, response: Dict[str, Any], formatted_data: List[Dict]) -> str:
        """Create a one-sentence summary answering the user's question"""
        meta = response.get("metadata", {})
        extracted_params = meta.get('extracted_parameters', {})
        original_query = meta.get('original_query', '')
        
        property_code = extracted_params.get('property_code', 'data')
        location = self._get_location_name(extracted_params.get('location_code', ''))
        
        if formatted_data:
            sensor = formatted_data[0]
            value = sensor['latest_value']
            unit = sensor['unit']
            time = sensor['latest_time']
            
            # Parse time to be more readable
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(time.replace('Z', '+00:00'))
                time_str = dt.strftime('%B %d, %Y at %H:%M UTC')
            except:
                time_str = time
            
            return f"**RESULT:** The latest {property_code} at {location} was **{value} {unit}** on {time_str}."
        
        return f"RESULT: Found {property_code} data from {location}."

    def _format_query_details(self, response: Dict[str, Any]) -> str:
        """Format detailed information about the query processing"""
        meta = response.get("metadata", {})
        extracted_params = meta.get('extracted_parameters', {})
        
        location_code = extracted_params.get('location_code', 'Unknown')
        device_category = extracted_params.get('device_category', 'Unknown')
        property_code = extracted_params.get('property_code', 'Unknown')
        
        # Get device information from successful API call
        device_info = self._get_device_info(response)
        location_name = self._get_location_name(location_code)
        
        lines = [
            "QUERY DETAILS:",
            "=" * 60,
            f"Location: {location_name} ({location_code})",
            f"Device Category: {device_category}",
            f"Property: {property_code}",
            f"Time Range: {extracted_params.get('start_time', 'Unknown')} to {extracted_params.get('end_time', 'Unknown')}",
        ]
        
        if device_info:
            lines.extend([
                "",
                "DEVICE USED:",
                f"   Name: {device_info['name']}",
                f"   Code: {device_info['code']}",
                f"   Type: {device_info['type']}"
            ])
        
        execution_time = meta.get('total_execution_time', 'Unknown')
        lines.append(f"Processing Time: {execution_time}s")
        
        return "\n".join(lines)

    def _format_api_calls(self, response: Dict[str, Any]) -> str:
        """Format API calls made during the query"""
        if "raw_api_responses" not in response:
            return ""
        
        raw_responses = response["raw_api_responses"]
        lines = [
            "",
            "## OCEAN NETWORKS CANADA API CALLS:",
            ""
        ]
        
        # Show devices API call
        if "devices_request" in raw_responses:
            devices_req = raw_responses["devices_request"]
            if "_debug_info" in devices_req:
                debug_info = devices_req["_debug_info"]
                params = debug_info.get('params', {})
                clean_params = {k: v for k, v in params.items() if k != 'token'}
                
                lines.extend([
                    "**1. Get Available Devices:**",
                    f"   - **URL:** `{debug_info.get('url', 'Unknown')}`",
                    f"   - **Parameters:** `{clean_params}`",
                    f"   - **Response:** Found {len(devices_req.get('data', []))} available devices",
                    ""
                ])
        
        # Show sensor data API calls
        if "scalar_data_requests" in raw_responses:
            for i, req in enumerate(raw_responses["scalar_data_requests"], 2):
                response_data = req.get('response', {})
                sensor_data = response_data.get('sensorData', [])
                device_name = req.get('device_name', 'Unknown')
                
                lines.extend([
                    f"**{i}. Get Sensor Data - {device_name}:**",
                ])
                
                if "_debug_info" in response_data:
                    debug_info = response_data["_debug_info"]
                    params = debug_info.get('params', {})
                    clean_params = {k: v for k, v in params.items() if k != 'token'}
                    
                    lines.extend([
                        f"   - **URL:** `{debug_info.get('url', 'Unknown')}`",
                        f"   - **Parameters:** `{clean_params}`",
                    ])
                
                if sensor_data:
                    lines.append(f"   - **Response:** SUCCESS - Found {len(sensor_data)} sensors with data")
                    if response["status"] == "success":
                        break  # Stop showing after successful device
                else:
                    lines.append("   - **Response:** NO DATA - No data found")
        
        return "\n".join(lines)

    def _format_suggestions(self, response: Dict[str, Any]) -> str:
        """Format suggestions for other queries the user can make"""
        meta = response.get("metadata", {})
        extracted_params = meta.get('extracted_parameters', {})
        location_code = extracted_params.get('location_code', 'CBYIP')
        
        # Get available properties from the extractor
        available_devices = list(self.extractor.location_devices.get(location_code, []))
        
        # Get some example properties from different devices
        suggestions = []
        device_examples = {
            'CTD': ['temperature', 'salinity', 'depth', 'pressure'],
            'PHSENSOR': ['pH'],
            'OXYSENSOR': ['oxygen'],
            'METSTN': ['wind speed', 'air temperature']
        }
        
        for device in ['CTD', 'PHSENSOR', 'OXYSENSOR']:
            if device in available_devices:
                properties = device_examples.get(device, [])
                if properties:
                    suggestions.extend(properties[:2])  # Add first 2 properties
        
        location_name = self._get_location_name(location_code)
        
        lines = [
            "OTHER DATA YOU CAN QUERY:",
            "=" * 60,
            f"Try asking about other properties at {location_name}:",
        ]
        
        for prop in suggestions[:6]:  # Show max 6 suggestions
            lines.append(f"   • \"What is the {prop} in Cambridge Bay?\"")
        
        lines.extend([
            "",
            "Available device categories: " + ", ".join(available_devices[:8]) + ("..." if len(available_devices) > 8 else "")
        ])
        
        return "\n".join(lines)

    def _get_location_name(self, location_code: str) -> str:
        """Convert location code to human readable name"""
        location_names = {
            'CBYIP': 'Cambridge Bay',
            'CBYSP': 'Cambridge Bay Shore',
            'CBYSS': 'Cambridge Bay Shore Station',
            'CBYSS.M1': 'Cambridge Bay Weather Station 1',
            'CBYSS.M2': 'Cambridge Bay Weather Station 2'
        }
        return location_names.get(location_code, location_code)

    def _format_device_discovery_response(self, response: Dict[str, Any]) -> str:
        """
        Format device discovery response with enhanced information
        
        Args:
            response: Device discovery response dictionary
            
        Returns:
            Formatted device discovery response string
        """
        devices = response["data"]
        metadata = response.get("metadata", {})
        
        if not devices:
            location_code = metadata.get("location_code", "Unknown")
            device_category = metadata.get("device_category")
            property_code = metadata.get("property_code")
            
            lines = ["🔍 **DEVICE DISCOVERY RESULT**", ""]
            lines.append("❌ **No devices found**")
            lines.append("")
            lines.append("**Search criteria:**")
            lines.append(f"  • Location: {location_code}")
            if device_category:
                lines.append(f"  • Device type: {device_category}")
            if property_code:
                lines.append(f"  • Property: {property_code}")
            
            lines.extend(["", "**Suggestions:**"])
            lines.append("  • Try searching without specific device type filters")
            lines.append("  • Check if the location code is correct")
            lines.append("  • Try 'What devices are at Cambridge Bay?' for broader search")
            
            return "\n".join(lines)
        
        # Group devices by location and category
        location_groups = {}
        category_totals = {}
        
        for device in devices:
            location_info = device.get('_location_info', {})
            location_code = location_info.get('locationCode', 'Unknown')
            location_name = location_info.get('locationName', location_code)
            category = device.get('deviceCategoryCode', 'Unknown')
            
            if location_code not in location_groups:
                location_groups[location_code] = {
                    'name': location_name,
                    'devices': {},
                    'total': 0
                }
            
            if category not in location_groups[location_code]['devices']:
                location_groups[location_code]['devices'][category] = []
            
            location_groups[location_code]['devices'][category].append(device)
            location_groups[location_code]['total'] += 1
            
            # Track category totals
            category_totals[category] = category_totals.get(category, 0) + 1
        
        # Build response
        lines = ["🔍 **DEVICE DISCOVERY RESULT**", ""]
        
        # Summary
        total_devices = len(devices)
        total_locations = len(location_groups)
        total_categories = len(category_totals)
        
        lines.append(f"✅ **Found {total_devices} device{'s' if total_devices != 1 else ''} across {total_locations} location{'s' if total_locations != 1 else ''} ({total_categories} device type{'s' if total_categories != 1 else ''})**")
        lines.append("")
        
        # Category summary
        lines.append("📊 **Device Types Summary:**")
        for category, count in sorted(category_totals.items()):
            device_type_name = self._get_device_type_description(category)
            lines.append(f"  • **{category}** ({device_type_name}): {count} device{'s' if count != 1 else ''}")
        lines.append("")
        
        # Detailed breakdown by location
        lines.append("📍 **By Location:**")
        
        for location_code, location_data in sorted(location_groups.items()):
            location_name = location_data['name']
            device_count = location_data['total']
            
            lines.append(f"\n**{location_name} ({location_code})** - {device_count} device{'s' if device_count != 1 else ''}")
            
            for category, category_devices in sorted(location_data['devices'].items()):
                device_type_name = self._get_device_type_description(category)
                lines.append(f"  🔧 **{category}** ({device_type_name}) - {len(category_devices)} device{'s' if len(category_devices) != 1 else ''}")
                
                # Show up to 3 devices per category per location
                for i, device in enumerate(category_devices[:3]):
                    device_name = device.get('deviceName', 'Unknown Device')
                    device_code = device.get('deviceCode', 'N/A')
                    supported_props = device.get('_supported_properties', [])
                    
                    lines.append(f"    • {device_name}")
                    lines.append(f"      Code: {device_code}")
                    if supported_props:
                        props_display = ", ".join(supported_props[:4])
                        if len(supported_props) > 4:
                            props_display += f", +{len(supported_props) - 4} more"
                        lines.append(f"      Measures: {props_display}")
                    
                if len(category_devices) > 3:
                    lines.append(f"    ... and {len(category_devices) - 3} more {category} device{'s' if len(category_devices) - 3 != 1 else ''}")
        
        # Add helpful footer
        lines.extend(["", "💡 **Next Steps:**"])
        lines.append("  • Ask for specific measurements: 'What is the temperature at Cambridge Bay?'")
        lines.append("  • Get device details: 'Tell me about CTD sensors at CBYIP'")
        lines.append("  • Download data: 'Download temperature data from Cambridge Bay'")
        
        # Add search metadata
        query_info = []
        if metadata.get("device_category"):
            query_info.append(f"Device type: {metadata['device_category']}")
        if metadata.get("property_code"):
            query_info.append(f"Property: {metadata['property_code']}")
        if metadata.get("execution_time"):
            query_info.append(f"Search time: {metadata['execution_time']}s")
        
        if query_info:
            lines.extend(["", f"ℹ️  Search details: {' • '.join(query_info)}"])
        
        return "\n".join(lines)

    def _get_device_type_description(self, device_category: str) -> str:
        """
        Get human-readable description for device category
        
        Args:
            device_category: Device category code
            
        Returns:
            Human-readable device description
        """
        descriptions = {
            'CTD': 'Conductivity-Temperature-Depth sensor',
            'HYDROPHONE': 'Underwater acoustic sensor',
            'OXYSENSOR': 'Dissolved oxygen sensor',
            'PHSENSOR': 'pH level sensor',
            'METSTN': 'Meteorological weather station',
            'ICEPROFILER': 'Ice-profiling sonar',
            'ADCP1200KHZ': 'Acoustic Doppler Current Profiler (1200kHz)',
            'BARPRESS': 'Barometric pressure sensor',
            'FLUOROMETER': 'Fluorescence measurement sensor',
            'FLNTU': 'Fluorescence and turbidity sensor',
            'TURBIDITYMETER': 'Water turbidity sensor',
            'RADIOMETER': 'Solar irradiance sensor',
            'NITRATESENSOR': 'Nitrate concentration sensor',
            'WETLABS_WQM': 'Water Quality Monitor',
            'VIDEOCAM': 'Underwater video camera',
            'CAMLIGHTS': 'Camera with lighting system',
            'PLANKTONSAMPLER': 'Plankton sampling device',
            'ACOUSTICRECEIVER': 'Acoustic tag receiver',
            'AISRECEIVER': 'Automatic Identification System receiver',
            'ICE_BUOY': 'Ice monitoring buoy',
            'ADCP': 'Acoustic Doppler Current Profiler',
            'CAMERA': 'Underwater camera system'
        }
        return descriptions.get(device_category, 'Ocean monitoring device')

    def _get_device_info(self, response: Dict[str, Any]) -> Dict[str, str]:
        """Extract device information from successful API response"""
        if "raw_api_responses" not in response:
            return {}
        
        # Find the successful device from scalar data requests
        scalar_requests = response["raw_api_responses"].get("scalar_data_requests", [])
        for req in scalar_requests:
            response_data = req.get('response', {})
            if response_data.get('sensorData'):
                return {
                    'name': req.get('device_name', 'Unknown'),
                    'code': req.get('device_code', 'Unknown'),
                    'type': req.get('device_category', 'Unknown')
                }
        
        return {}

    def get_available_options(self) -> Dict[str, Any]:
        """Get all available options for reference"""
        return self.extractor.get_available_options()

    def close(self):
        """Clean up resources"""
        self.api_client.close()
        logger.info("Ocean Query System closed")

    def process_data_download_query(self, query: str, include_metadata: bool = True, 
                                   row_limit: int = 1000) -> Dict[str, Any]:
        """
        Process a data download query for CSV export or data product downloads
        
        Args:
            query: Natural language query about data downloads
            include_metadata: Whether to include processing metadata
            row_limit: Maximum rows for data downloads
            
        Returns:
            Response with download information and file paths
        """
        start_time = time.time()
        logger.info(f"Processing data download query: '{query}'")
        
        # Step 1: Extract parameters from query
        logger.info("Step 1: Extracting parameters...")
        extraction_result = self.extractor.extract_parameters(query)
        
        if extraction_result["status"] != "success":
            return {
                "status": "error",
                "stage": "parameter_extraction",
                "message": extraction_result["message"],
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "data_download",
                    "extraction_result": extraction_result,
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        params = extraction_result["parameters"]
        logger.info(f"Extracted parameters for data download: {params}")
        
        # Step 2: Determine download type and process
        try:
            location_code = params["location_code"]
            device_category = params.get("device_category")
            property_code = params.get("property_code")
            temporal_reference = params.get("temporal_reference")
            
            # Check for specific date range or instance
            download_type = self._determine_download_type(query, temporal_reference)
            
            if download_type == "csv_export":
                return self._process_csv_download(query, params, include_metadata, start_time)
            elif download_type == "data_product":
                return self._process_data_product_download(query, params, include_metadata, start_time)
            else:
                # Default to CSV export for data downloads
                return self._process_csv_download(query, params, include_metadata, start_time)
                
        except Exception as e:
            logger.error(f"Data download processing failed: {e}")
            return {
                "status": "error",
                "stage": "data_download_processing",
                "message": f"Data download processing failed: {str(e)}",
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "data_download",
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
    
    def _determine_download_type(self, query: str, temporal_reference: str) -> str:
        """Determine the type of download requested"""
        query_lower = query.lower()
        
        # CSV export indicators
        csv_keywords = ['csv', 'export', 'download data', 'spreadsheet', 'excel']
        if any(keyword in query_lower for keyword in csv_keywords):
            return "csv_export"
        
        # Data product indicators
        product_keywords = ['plot', 'graph', 'visualization', 'png', 'pdf', 'image']
        if any(keyword in query_lower for keyword in product_keywords):
            return "data_product"
        
        # Default to CSV for data downloads
        return "csv_export"
    
    def _process_csv_download(self, query: str, params: Dict[str, Any], 
                             include_metadata: bool, start_time: float) -> Dict[str, Any]:
        """Process CSV data download request using query routing"""
        try:
            location_code = params["location_code"]
            device_category = params.get("device_category", "CTD")  # Default to CTD
            temporal_reference = params.get("temporal_reference")
            
            # Parse temporal reference for date range
            date_from, date_to = self._parse_temporal_reference(temporal_reference)
            
            if not date_from or not date_to:
                # Default to last 24 hours if no specific dates
                from datetime import datetime, timedelta
                date_to = datetime.utcnow()
                date_from = date_to - timedelta(days=1)
                date_from = date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                date_to = date_to.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            
            # Use advanced data downloader for CSV export
            logger.info(f"Downloading CSV data for {device_category} at {location_code}")
            download_result = self.data_downloader.download_csv_data(
                location_code=location_code,
                device_category=device_category,
                date_from=date_from,
                date_to=date_to,
                output_dir="csv_downloads",
                quality_control=True,
                resample="none"
            )
            
            if download_result['status'] == 'success':
                return {
                    "status": "success",
                    "message": f"CSV data downloaded successfully",
                    "data": download_result,
                    "download_info": {
                        "csv_files": download_result.get('csv_files', []),
                        "download_directory": download_result.get('output_directory', 'csv_downloads'),
                        "file_count": len(download_result.get('csv_files', [])),
                        "location_code": location_code,
                        "device_category": device_category,
                        "date_range": f"{date_from} to {date_to}"
                    },
                    "metadata": {
                        "query": query,
                        "query_type": "csv_download",
                        "download_requested": True,
                        "execution_time": time.time() - start_time
                    } if include_metadata else None
                }
            else:
                return {
                    "status": "error",
                    "message": f"CSV download failed: {download_result.get('message', 'Unknown error')}",
                    "data": None,
                    "metadata": {
                        "query": query,
                        "query_type": "csv_download",
                        "download_requested": True,
                        "execution_time": time.time() - start_time
                    } if include_metadata else None
                }
                
        except Exception as e:
            logger.error(f"CSV download failed: {e}")
            return {
                "status": "error",
                "stage": "csv_download",
                "message": f"CSV download failed: {str(e)}",
                "data": None,
                "metadata": {
                    "query": query,
                    "query_type": "csv_download",
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
    
    def _parse_temporal_reference(self, temporal_reference: str) -> tuple:
        """Parse temporal reference into date from and date to"""
        if not temporal_reference or temporal_reference == "latest":
            return None, None
        
        try:
            from datetime import datetime, timedelta
            import re
            
            # Handle specific date patterns
            if "to" in temporal_reference or "-" in temporal_reference:
                # Date range: "2023-01-01 to 2023-01-02" or "2023-01-01 - 2023-01-02"
                parts = re.split(r'\s+to\s+|\s*-\s*', temporal_reference)
                if len(parts) == 2:
                    date_from = parts[0].strip()
                    date_to = parts[1].strip()
                    
                    # Convert to ISO format if needed
                    if not date_from.endswith('Z'):
                        date_from = f"{date_from}T00:00:00.000Z"
                    if not date_to.endswith('Z'):
                        date_to = f"{date_to}T23:59:59.999Z"
                    
                    return date_from, date_to
            
            # Single date - use as start date with 24-hour range
            elif len(temporal_reference) >= 10:  # Looks like a date
                base_date = temporal_reference.strip()
                
                # Handle future dates by using a reasonable date range for demo
                if "2025" in base_date:
                    logger.warning(f"Future date {base_date} detected, using 2023 data instead for demo")
                    base_date = base_date.replace("2025", "2023")
                
                if not base_date.endswith('Z'):
                    date_from = f"{base_date}T00:00:00.000Z"
                    date_to = f"{base_date}T23:59:59.999Z"
                else:
                    # Parse and add 24 hours
                    dt = datetime.fromisoformat(base_date.replace('Z', '+00:00'))
                    date_from = base_date
                    date_to = (dt + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
                return date_from, date_to
            
            return None, None
            
        except Exception as e:
            logger.warning(f"Could not parse temporal reference '{temporal_reference}': {e}")
            return None, None


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ocean Query System - Natural Language to ONC API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocean_query_system.py "What is the temperature in Cambridge Bay today?"
  python ocean_query_system.py "Show me wind speed at the weather station"
  python ocean_query_system.py --latest "Cambridge Bay salinity" --hours 12
  python ocean_query_system.py --interactive
        """
    )
    
    parser.add_argument('query', nargs='*', help='Natural language query')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--latest', action='store_true', help='Get latest data only')
    parser.add_argument('--hours', type=int, default=24, help='Hours back for latest data')
    parser.add_argument('--json', action='store_true', help='Output raw JSON instead of formatted text')
    parser.add_argument('--no-metadata', action='store_true', help='Exclude metadata from output')
    parser.add_argument('--raw-data', action='store_true', help='Include full raw API responses and debug info')
    parser.add_argument('--show-api-calls', action='store_true', help='Show clean API calls made (URLs and parameters)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        system = OceanQuerySystem()
        
        if args.interactive:
            # Interactive mode
            print("Ocean Query System - Interactive Mode")
            print("=" * 60)
            print("Examples:")
            print("  - What is the temperature in Cambridge Bay?")
            print("  - Show me wind speed at the weather station")
            print("  - Cambridge Bay salinity data from yesterday")
            print("\nCommands:")
            print("  - 'options' - show available locations/devices/properties")
            print("  - 'quit' or 'exit' - exit the system")
            print("\nAPI calls will be shown for each query")
            print("Type your queries below:\n")
            
            while True:
                try:
                    query = input("Query: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if query.lower() == 'options':
                        options = system.get_available_options()
                        print("\nAvailable Options:")
                        print(json.dumps(options, indent=2))
                        print()
                        continue
                    
                    if not query:
                        continue
                    
                    print("\nProcessing...")
                    
                    # Process query
                    response = system.process_query(query, include_metadata=not args.no_metadata)
                    
                    if args.json:
                        print(json.dumps(response, indent=2, default=str))
                    else:
                        # In interactive mode, always show API calls for educational purposes
                        print(system.format_response_for_display(
                            response, 
                            include_raw_data=args.raw_data,
                            show_api_calls=True
                        ))
                    
                    print("\n" + "=" * 60 + "\n")
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"ERROR: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
        
        elif args.query:
            # Single query mode
            query = " ".join(args.query)
            
            if args.latest:
                response = system.get_latest_data(query, args.hours)
            else:
                response = system.process_query(query, include_metadata=not args.no_metadata)
            
            if args.json:
                print(json.dumps(response, indent=2, default=str))
            else:
                print(system.format_response_for_display(
                    response, 
                    include_raw_data=args.raw_data,
                    show_api_calls=args.show_api_calls
                ))
        
        else:
            # Show help if no arguments
            parser.print_help()
        
        system.close()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"SYSTEM ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()