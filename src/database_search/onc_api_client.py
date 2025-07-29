#!/usr/bin/env python3
"""
ONC API Client Module
Handles all Ocean Networks Canada API interactions
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ONCAPIClient:
    """Client for Ocean Networks Canada API"""
    
    def __init__(self, token: str = None):
        """
        Initialize ONC API client
        
        Args:
            token: ONC API token. If None, will try to get from environment
        """
        # Default to the token from existing code, but allow override
        self.token = token or "b77b663d-e93b-40a3-a653-dfccb4a1b0cb"
        self.base_url = "https://data.oceannetworks.ca/api"
        self.timeout = 30
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Request tracking
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum 100ms between requests

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the ONC API with error handling and rate limiting
        
        Args:
            endpoint: API endpoint (e.g., 'devices', 'scalardata/device')
            params: Request parameters
            
        Returns:
            API response as dictionary with debug info
        """
        # Add token to parameters
        params = {**params, 'token': self.token}
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            logger.debug(f"Making request to {endpoint} with params: {params}")
            start_time = time.time()
            
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            request_time = time.time() - start_time
            self.last_request_time = time.time()
            
            logger.debug(f"Request completed in {request_time:.2f}s with status {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API request failed: HTTP {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {
                    "error": f"HTTP {response.status_code}",
                    "message": response.text,
                    "_debug_info": {
                        "url": url,
                        "params": params,
                        "status_code": response.status_code,
                        "request_time": request_time
                    }
                }
            
            response_data = response.json()
            
            # Add debug information to successful responses
            debug_info = {
                "url": url,
                "params": params,
                "status_code": response.status_code,
                "request_time": request_time,
                "response_size": len(response.text)
            }
            
            # If response is a list, wrap it in a dict to add debug info
            if isinstance(response_data, list):
                return {
                    "data": response_data,
                    "_debug_info": debug_info
                }
            else:
                response_data["_debug_info"] = debug_info
                return response_data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout}s")
            return {
                "error": "timeout", 
                "message": "Request timed out",
                "_debug_info": {
                    "url": url,
                    "params": params,
                    "timeout": self.timeout
                }
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {
                "error": "request_failed", 
                "message": str(e),
                "_debug_info": {
                    "url": url,
                    "params": params,
                    "exception": str(e)
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return {
                "error": "json_parse_error", 
                "message": str(e),
                "_debug_info": {
                    "url": url,
                    "params": params,
                    "raw_response": response.text[:1000]  # First 1000 chars
                }
            }

    def get_devices(self, location_code: str, device_category: str = None) -> List[Dict[str, Any]]:
        """
        Get devices for a location, optionally filtered by device category
        
        Args:
            location_code: ONC location code (e.g., 'CBYIP')
            device_category: Optional device category filter (e.g., 'CTD')
            
        Returns:
            List of device information dictionaries
        """
        params = {'locationCode': location_code}
        if device_category:
            params['deviceCategoryCode'] = device_category
        
        response = self._make_request('devices', params)
        
        if 'error' in response:
            logger.error(f"Failed to get devices: {response}")
            return []
        
        # Extract devices from response data
        if isinstance(response, dict) and 'data' in response:
            devices = response['data'] if isinstance(response['data'], list) else []
        elif isinstance(response, list):
            devices = response
        else:
            devices = []
        
        # Filter by device category if specified and not already filtered by API
        if device_category and not any('deviceCategoryCode' in p for p in [params]):
            devices = [d for d in devices if d.get('deviceCategoryCode') == device_category]
        
        logger.info(f"Found {len(devices)} devices for location {location_code}")
        return devices

    def get_scalar_data(self, device_code: str, property_code: str = None, 
                       date_from: str = None, date_to: str = None, 
                       row_limit: int = 100) -> Dict[str, Any]:
        """
        Get scalar data from a device
        
        Args:
            device_code: ONC device code
            property_code: Optional property code filter
            date_from: Start date (ISO format with .000Z)
            date_to: End date (ISO format with .000Z)
            row_limit: Maximum number of rows
            
        Returns:
            Raw scalar data response
        """
        params = {
            'deviceCode': device_code,
            'rowLimit': row_limit
        }
        
        # Handle date formatting for ONC API
        if date_from:
            if date_from.endswith('.000Z'):
                params['dateFrom'] = date_from
            elif 'T' in date_from:
                # Remove any existing Z and add .000Z
                params['dateFrom'] = date_from.rstrip('Z') + '.000Z'
            else:
                params['dateFrom'] = date_from + 'T00:00:00.000Z'
        
        if date_to:
            if date_to.endswith('.000Z'):
                params['dateTo'] = date_to
            elif 'T' in date_to:
                # Remove any existing Z and add .000Z
                params['dateTo'] = date_to.rstrip('Z') + '.000Z'
            else:
                params['dateTo'] = date_to + 'T00:00:00.000Z'
        
        # Don't include property filter - let's get all data and filter later
        # The API seems to be rejecting sensorCategoryCodes parameter
        
        response = self._make_request('scalardata/device', params)
        
        if 'error' in response:
            logger.error(f"Failed to get scalar data: {response}")
            return {"sensorData": []}
        
        return response

    def get_raw_data(self, device_code: str, date_from: str = None, 
                    date_to: str = None, row_limit: int = 100, 
                    output_format: str = "object") -> Dict[str, Any]:
        """
        Get raw data from a device
        
        Args:
            device_code: ONC device code
            date_from: Start date (ISO format)
            date_to: End date (ISO format) 
            row_limit: Maximum number of rows
            output_format: Output format ('object', 'json', etc.)
            
        Returns:
            Raw data response
        """
        params = {
            'deviceCode': device_code,
            'rowLimit': row_limit,
            'outputFormat': output_format
        }
        
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
        
        response = self._make_request('rawdata/device', params)
        
        if 'error' in response:
            logger.error(f"Failed to get raw data: {response}")
            return {"data": []}
        
        return response

    def search_data_range(self, location_code: str, device_category: str, 
                         property_code: str, date_from: str, date_to: str, 
                         row_limit: int = 1000, max_devices: int = None,
                         parallel: bool = False) -> Dict[str, Any]:
        """
        Enhanced method to search for data ranges with multiple options
        
        Args:
            location_code: ONC location code
            device_category: ONC device category code
            property_code: ONC property code
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            row_limit: Maximum rows per device (default: 1000)
            max_devices: Maximum number of devices to query (None = all)
            parallel: Whether to query devices in parallel
            
        Returns:
            Structured response with aggregated data from multiple devices
        """
        start_time = time.time()
        
        # Get devices
        devices_response = self._make_request('devices', {
            'locationCode': location_code,
            'deviceCategoryCode': device_category
        })
        
        if isinstance(devices_response, dict) and 'error' in devices_response:
            devices = []
        elif isinstance(devices_response, list):
            devices = devices_response
        elif isinstance(devices_response, dict) and 'data' in devices_response:
            devices = devices_response['data']
        else:
            devices = []
            
        if not devices:
            return {
                "status": "error",
                "message": f"No {device_category} devices found at location {location_code}",
                "data": [],
                "raw_api_responses": {"devices_request": devices_response}
            }
        
        # Limit devices if specified
        if max_devices and len(devices) > max_devices:
            devices = devices[:max_devices]
        
        if parallel:
            return self._search_devices_parallel(devices, property_code, date_from, date_to, row_limit, devices_response)
        else:
            return self._search_devices_sequential(devices, property_code, date_from, date_to, row_limit, devices_response)

    def search_data(self, location_code: str, device_category: str, 
                   property_code: str, date_from: str, date_to: str, 
                   row_limit: int = 100) -> Dict[str, Any]:
        """
        High-level method to search for data with extracted parameters
        
        Args:
            location_code: ONC location code
            device_category: ONC device category code
            property_code: ONC property code
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            row_limit: Maximum number of rows
            
        Returns:
            Structured response with data and metadata
        """
        start_time = time.time()
        
        # Step 1: Get devices
        devices_response = self._make_request('devices', {
            'locationCode': location_code,
            'deviceCategoryCode': device_category
        })
        
        # Handle devices response properly
        if isinstance(devices_response, dict) and 'error' in devices_response:
            devices = []
        elif isinstance(devices_response, list):
            devices = devices_response
        elif isinstance(devices_response, dict) and 'data' in devices_response:
            # Response was a list wrapped in dict with debug info
            devices = devices_response['data']
        else:
            devices = []
            
        if not devices:
            return {
                "status": "error",
                "message": f"No {device_category} devices found at location {location_code}",
                "data": [],
                "raw_api_responses": {
                    "devices_request": devices_response
                }
            }
        
        # Step 2: Try each device until we find data
        all_sensor_data = []
        successful_device = None
        all_api_responses = {
            "devices_request": devices_response,
            "scalar_data_requests": []
        }
        
        for device in devices:
            device_code = device['deviceCode']
            device_name = device.get('deviceName', device_code)
            
            logger.info(f"Trying device: {device_name} ({device_code})")
            
            # Get scalar data for this device
            scalar_response = self.get_scalar_data(
                device_code=device_code,
                property_code=property_code,
                date_from=date_from,
                date_to=date_to,
                row_limit=row_limit
            )
            
            # Store the raw API response for debugging
            all_api_responses["scalar_data_requests"].append({
                "device_code": device_code,
                "device_name": device_name,
                "response": scalar_response
            })
            
            sensor_data_list = scalar_response.get('sensorData', [])
            
            # Ensure sensor_data_list is not None
            if sensor_data_list is None:
                sensor_data_list = []
            
            # Filter for the specific property we want
            matching_sensors = []
            for sensor in sensor_data_list:
                if sensor and sensor.get('propertyCode') == property_code:
                    matching_sensors.append(sensor)
            
            if matching_sensors:
                all_sensor_data.extend(matching_sensors)
                successful_device = device
                logger.info(f"Found {len(matching_sensors)} sensors with {property_code} data")
                break
            else:
                logger.info(f"No {property_code} data found from {device_name}")
        
        # Step 3: Format response
        total_time = time.time() - start_time
        
        if not all_sensor_data:
            return {
                "status": "no_data",
                "message": f"No {property_code} data found from any {device_category} devices at {location_code}",
                "data": [],
                "metadata": {
                    "devices_checked": len(devices),
                    "execution_time": round(total_time, 2)
                },
                "raw_api_responses": all_api_responses
            }
        
        return {
            "status": "success",
            "message": f"Found {len(all_sensor_data)} sensors with {property_code} data",
            "data": all_sensor_data,
            "metadata": {
                "location_code": location_code,
                "device_category": device_category,
                "property_code": property_code,
                "successful_device": successful_device,
                "devices_checked": len(devices),
                "execution_time": round(total_time, 2),
                "date_range": {
                    "from": date_from,
                    "to": date_to
                }
            },
            "raw_api_responses": all_api_responses
        }

    def get_latest_data(self, location_code: str, device_category: str, 
                       property_code: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get the most recent data for a property
        
        Args:
            location_code: ONC location code
            device_category: ONC device category code  
            property_code: ONC property code
            hours_back: How many hours back to search
            
        Returns:
            Latest data response
        """
        # Calculate time range - go back further to find data
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours_back)
        
        date_from = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        date_to = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        return self.search_data(
            location_code=location_code,
            device_category=device_category,
            property_code=property_code,
            date_from=date_from,
            date_to=date_to,
            row_limit=10  # Just get recent data
        )

    def format_sensor_data(self, sensor_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format sensor data into a more readable structure
        
        Args:
            sensor_data: Raw sensor data from API
            
        Returns:
            Formatted sensor data
        """
        formatted = []
        
        for sensor in sensor_data:
            sensor_name = sensor.get("sensorName", "Unknown Sensor")
            property_code = sensor.get("propertyCode", "Unknown Property")
            unit = sensor.get("unitOfMeasure", "")
            
            data = sensor.get("data", {})
            values = data.get("values", [])
            sample_times = data.get("sampleTimes", [])
            qaqc_flags = data.get("qaqcFlags", [])
            
            if not values or not sample_times:
                continue
            
            # Get most recent reading
            latest_value = values[-1] if values else None
            latest_time = sample_times[-1] if sample_times else None
            latest_qaqc = qaqc_flags[-1] if qaqc_flags else None
            
            formatted_entry = {
                "sensor_name": sensor_name,
                "property_code": property_code,
                "unit": unit,
                "latest_value": latest_value,
                "latest_time": latest_time,
                "qaqc_flag": latest_qaqc,
                "qaqc_status": "Passed" if latest_qaqc == 0 else "Check Required",
                "total_readings": len(values),
                "all_values": values,
                "all_times": sample_times
            }
            
            formatted.append(formatted_entry)
        
        return formatted

    def _search_devices_sequential(self, devices, property_code, date_from, date_to, row_limit, devices_response):
        """Search devices sequentially (existing logic)"""
        all_sensor_data = []
        successful_devices = []
        all_api_responses = {
            "devices_request": devices_response,
            "scalar_data_requests": []
        }
        
        for device in devices:
            device_code = device['deviceCode']
            device_name = device.get('deviceName', device_code)
            
            logger.info(f"Trying device: {device_name} ({device_code})")
            
            scalar_response = self.get_scalar_data(
                device_code=device_code,
                property_code=property_code,
                date_from=date_from,
                date_to=date_to,
                row_limit=row_limit
            )
            
            all_api_responses["scalar_data_requests"].append({
                "device_code": device_code,
                "device_name": device_name,
                "response": scalar_response
            })
            
            sensor_data_list = scalar_response.get('sensorData', []) or []
            
            matching_sensors = [
                sensor for sensor in sensor_data_list 
                if sensor and sensor.get('propertyCode') == property_code
            ]
            
            if matching_sensors:
                all_sensor_data.extend(matching_sensors)
                successful_devices.append(device)
                logger.info(f"Found {len(matching_sensors)} sensors with {property_code} data")
        
        return self._format_range_response(all_sensor_data, successful_devices, devices, all_api_responses, property_code)

    def _search_devices_parallel(self, devices, property_code, date_from, date_to, row_limit, devices_response):
        """Search devices in parallel using threading"""
        import concurrent.futures
        import threading
        
        all_sensor_data = []
        successful_devices = []
        all_api_responses = {
            "devices_request": devices_response,
            "scalar_data_requests": []
        }
        lock = threading.Lock()
        
        def query_device(device):
            device_code = device['deviceCode']
            device_name = device.get('deviceName', device_code)
            
            scalar_response = self.get_scalar_data(
                device_code=device_code,
                property_code=property_code,
                date_from=date_from,
                date_to=date_to,
                row_limit=row_limit
            )
            
            with lock:
                all_api_responses["scalar_data_requests"].append({
                    "device_code": device_code,
                    "device_name": device_name,
                    "response": scalar_response
                })
            
            sensor_data_list = scalar_response.get('sensorData', []) or []
            matching_sensors = [
                sensor for sensor in sensor_data_list 
                if sensor and sensor.get('propertyCode') == property_code
            ]
            
            if matching_sensors:
                with lock:
                    all_sensor_data.extend(matching_sensors)
                    successful_devices.append(device)
                logger.info(f"Found {len(matching_sensors)} sensors with {property_code} data from {device_name}")
        
        # Execute parallel queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_device, device) for device in devices]
            concurrent.futures.wait(futures)
        
        return self._format_range_response(all_sensor_data, successful_devices, devices, all_api_responses, property_code)

    def _format_range_response(self, all_sensor_data, successful_devices, all_devices, all_api_responses, property_code):
        """Format the response for range queries"""
        if not all_sensor_data:
            return {
                "status": "no_data",
                "message": f"No {property_code} data found from any devices",
                "data": [],
                "metadata": {
                    "devices_checked": len(all_devices),
                    "successful_devices": 0,
                    "total_sensors": 0
                },
                "raw_api_responses": all_api_responses
            }
        
        # Aggregate statistics
        total_readings = sum(len(sensor.get('data', {}).get('values', [])) for sensor in all_sensor_data)
        
        return {
            "status": "success",
            "message": f"Found data from {len(successful_devices)} devices with {len(all_sensor_data)} sensors",
            "data": all_sensor_data,
            "metadata": {
                "devices_checked": len(all_devices),
                "successful_devices": len(successful_devices),
                "total_sensors": len(all_sensor_data),
                "total_readings": total_readings,
                "successful_device_codes": [d['deviceCode'] for d in successful_devices]
            },
            "raw_api_responses": all_api_responses
        }

    def get_paginated_data(self, location_code: str, device_category: str, 
                          property_code: str, date_from: str, date_to: str,
                          page_size: int = 500, max_pages: int = 10) -> Dict[str, Any]:
        """
        Get paginated data by splitting time ranges
        
        Args:
            location_code: ONC location code
            device_category: ONC device category code
            property_code: ONC property code
            date_from: Start date (ISO format)
            date_to: End date (ISO format)
            page_size: Rows per time chunk
            max_pages: Maximum number of time chunks
            
        Returns:
            Paginated response with data from multiple time windows
        """
        from datetime import datetime, timedelta
        
        start_dt = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
        
        total_duration = end_dt - start_dt
        chunk_duration = total_duration / max_pages
        
        all_data = []
        all_responses = []
        
        for i in range(max_pages):
            chunk_start = start_dt + (chunk_duration * i)
            chunk_end = start_dt + (chunk_duration * (i + 1))
            
            chunk_from = chunk_start.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            chunk_to = chunk_end.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            
            response = self.search_data_range(
                location_code=location_code,
                device_category=device_category,
                property_code=property_code,
                date_from=chunk_from,
                date_to=chunk_to,
                row_limit=page_size
            )
            
            all_responses.append({
                "chunk": i + 1,
                "time_range": {"from": chunk_from, "to": chunk_to},
                "response": response
            })
            
            if response.get('status') == 'success':
                all_data.extend(response.get('data', []))
        
        return {
            "status": "success" if all_data else "no_data",
            "message": f"Retrieved {len(all_data)} sensors across {max_pages} time chunks",
            "data": all_data,
            "metadata": {
                "pagination": {
                    "total_chunks": max_pages,
                    "chunk_duration_hours": chunk_duration.total_seconds() / 3600,
                    "page_size": page_size
                }
            },
            "raw_responses": all_responses
        }

    def find_cambridge_bay_devices(self, device_category: str = None, property_code: str = None) -> List[Dict[str, Any]]:
        """
        Find devices at Cambridge Bay locations with optional filtering
        
        Args:
            device_category: Optional device category filter (e.g., 'CTD', 'HYDROPHONE')
            property_code: Optional property code filter (e.g., 'seawatertemperature')
            
        Returns:
            List of device dictionaries with enhanced location information
        """
        # Accurate Cambridge Bay location to device mapping based on ONC API
        location_device_mapping = {
            'CBYIP': ['ACOUSTICRECEIVER', 'ADCP1200KHZ', 'CAMLIGHTS', 'CTD', 'FLNTU', 
                     'FLUOROMETER', 'HYDROPHONE', 'ICEPROFILER', 'NITRATESENSOR', 
                     'OXYSENSOR', 'PHSENSOR', 'PLANKTONSAMPLER', 'RADIOMETER', 
                     'TURBIDITYMETER', 'VIDEOCAM', 'WETLABS_WQM'],
            'CBYSP': ['ICE_BUOY'],
            'CBYSS': ['AISRECEIVER', 'BARPRESS', 'VIDEOCAM'],
            'CBYSS.M1': ['METSTN'],
            'CBYSS.M2': ['METSTN']
        }
        
        # Property to device category mapping for filtering
        property_device_mapping = {
            'seawatertemperature': ['CTD', 'OXYSENSOR', 'PHSENSOR', 'ADCP1200KHZ'],
            'salinity': ['CTD'],
            'pressure': ['CTD'],
            'conductivity': ['CTD'],
            'depth': ['CTD'],
            'oxygen': ['OXYSENSOR'],
            'ph': ['PHSENSOR'], 
            'soundpressurelevel': ['HYDROPHONE', 'ICEPROFILER'],
            'amperage': ['HYDROPHONE'],
            'batterycharge': ['HYDROPHONE'],
            'voltage': ['HYDROPHONE'],
            'internaltemperature': ['HYDROPHONE'],
            'airtemperature': ['METSTN'],
            'windspeed': ['METSTN'],
            'humidity': ['METSTN'],
            'absolutebarometricpressure': ['METSTN', 'BARPRESS'],
            'icedraft': ['ICEPROFILER'],
            'pingtime': ['ICEPROFILER']
        }
        
        all_devices = []
        locations_to_search = []
        
        # Determine which locations to search based on device category filter
        if device_category:
            # Only search locations that have the requested device category
            for location, devices in location_device_mapping.items():
                if device_category in devices:
                    locations_to_search.append(location)
            logger.info(f"Searching for {device_category} devices at locations: {locations_to_search}")
        else:
            # Search all Cambridge Bay locations
            locations_to_search = list(location_device_mapping.keys())
            logger.info(f"Searching all Cambridge Bay locations: {locations_to_search}")
        
        # If property filter is specified, further narrow down locations
        if property_code:
            compatible_device_categories = property_device_mapping.get(property_code, [])
            if compatible_device_categories:
                # Only search locations that have devices compatible with the property
                filtered_locations = []
                for location, available_devices in location_device_mapping.items():
                    if any(dev_cat in available_devices for dev_cat in compatible_device_categories):
                        filtered_locations.append(location)
                
                # Intersect with device category filtered locations
                locations_to_search = list(set(locations_to_search) & set(filtered_locations))
                logger.info(f"After property filter ({property_code}), searching locations: {locations_to_search}")
        
        if not locations_to_search:
            logger.info("No locations found matching the search criteria")
            return []
        
        for location_code in locations_to_search:
            try:
                available_devices_at_location = location_device_mapping.get(location_code, [])
                
                # If device_category is specified, only search for that category if it exists at this location
                search_device_category = None
                if device_category and device_category in available_devices_at_location:
                    search_device_category = device_category
                elif device_category and device_category not in available_devices_at_location:
                    # Skip this location since it doesn't have the requested device category
                    continue
                
                # Get devices for this location
                location_devices = self.get_devices(location_code, search_device_category)
                
                # Enhance each device with location info and filter by property if needed
                for device in location_devices:
                    # Add enhanced location information
                    device['_location_info'] = {
                        'locationCode': location_code,
                        'locationName': self._get_location_name(location_code),
                        'region': 'Cambridge Bay'
                    }
                    
                    # Filter by property code if specified
                    if property_code:
                        device_cat = device.get('deviceCategoryCode', '')
                        compatible_devices = property_device_mapping.get(property_code, [])
                        
                        if device_cat in compatible_devices:
                            device['_property_compatible'] = True
                            device['_supported_properties'] = self._get_device_properties(device_cat)
                            all_devices.append(device)
                        # Skip devices that don't support the requested property
                    else:
                        # No property filter, add all devices
                        device['_supported_properties'] = self._get_device_properties(
                            device.get('deviceCategoryCode', '')
                        )
                        all_devices.append(device)
                        
            except Exception as e:
                logger.warning(f"Failed to get devices for location {location_code}: {e}")
                continue
        
        # Sort devices by location and then by device category
        all_devices.sort(key=lambda x: (
            x.get('_location_info', {}).get('locationCode', ''),
            x.get('deviceCategoryCode', ''),
            x.get('deviceName', '')
        ))
        
        logger.info(f"Found {len(all_devices)} devices across Cambridge Bay locations")
        
        return all_devices

    def discover_cambridge_bay_data_products(self, device_category: str = None, 
                                           data_product_code: str = None) -> List[Dict[str, Any]]:
        """
        Discover available data products at Cambridge Bay locations
        
        Args:
            device_category: Optional device category filter
            data_product_code: Optional data product code filter
            
        Returns:
            List of available data products
        """
        cambridge_locations = ["CBYIP", "CBYSS.M1", "CBYSS.M2", "CBYDS", "CBYSP"]
        all_products = []
        
        for location_code in cambridge_locations:
            try:
                products = self.get_data_products(location_code, device_category, data_product_code)
                
                # Enhance with location info
                for product in products:
                    product['_location_info'] = {
                        'locationCode': location_code,
                        'locationName': self._get_location_name(location_code),
                        'region': 'Cambridge Bay'
                    }
                    all_products.append(product)
                    
            except Exception as e:
                logger.warning(f"Failed to get data products for {location_code}: {e}")
                continue
        
        return all_products

    def get_data_products(self, location_code: str, device_category: str = None, 
                         data_product_code: str = None) -> List[Dict[str, Any]]:
        """
        Get available data products for a location
        
        Args:
            location_code: ONC location code
            device_category: Optional device category filter
            data_product_code: Optional data product code filter
            
        Returns:
            List of data product information
        """
        params = {'locationCode': location_code}
        
        if device_category:
            params['deviceCategoryCode'] = device_category
        if data_product_code:
            params['dataProductCode'] = data_product_code
            
        response = self._make_request('dataProducts', params)
        
        if 'error' in response:
            logger.error(f"Failed to get data products: {response}")
            return []
            
        products = response if isinstance(response, list) else []
        logger.info(f"Found {len(products)} data products for {location_code}")
        
        return products

    def generate_download_status_url(self, request_id: str) -> str:
        """
        Generate URL to check download status for a data product request
        
        Args:
            request_id: Data product request ID
            
        Returns:
            URL to check download status
        """
        return f"{self.base_url}/dataProductDelivery?token={self.token}&dpRequestId={request_id}"

    def _get_location_name(self, location_code: str) -> str:
        """
        Convert location code to human-readable name
        
        Args:
            location_code: ONC location code
            
        Returns:
            Human-readable location name
        """
        location_names = {
            'CBYIP': 'Cambridge Bay Inshore',
            'CBYSS.M1': 'Cambridge Bay Shore Station Meteorological 1', 
            'CBYSS.M2': 'Cambridge Bay Shore Station Meteorological 2',
            'CBYDS': 'Cambridge Bay Deep Station',
            'CBYSP': 'Cambridge Bay Shallow Profiler',
            'CBYSS': 'Cambridge Bay Shore Station'
        }
        return location_names.get(location_code, location_code)

    def _get_device_properties(self, device_category: str) -> List[str]:
        """
        Get list of properties supported by a device category
        
        Args:
            device_category: Device category code
            
        Returns:
            List of supported property codes
        """
        # Extended device properties based on ONC API documentation
        device_properties = {
            "CTD": ["seawatertemperature", "salinity", "pressure", "depth", "conductivity"],
            "OXYSENSOR": ["oxygen", "seawatertemperature"],
            "PHSENSOR": ["ph", "seawatertemperature"],
            "METSTN": ["airtemperature", "windspeed", "humidity", "absolutebarometricpressure"],
            "ICEPROFILER": ["soundpressurelevel", "seawatertemperature", "icedraft", "pingtime"],
            "HYDROPHONE": ["soundpressurelevel", "amperage", "batterycharge", "voltage", "internaltemperature"],
            "ADCP1200KHZ": ["magneticheading", "pitch", "roll", "seawatertemperature", "soundspeed"],
            "BARPRESS": ["absolutebarometricpressure"],
            "FLUOROMETER": ["fluorescence", "seawatertemperature"],
            "FLNTU": ["turbidity", "fluorescence"],
            "TURBIDITYMETER": ["turbidity"],
            "RADIOMETER": ["irradiance"],
            "NITRATESENSOR": ["nitrate", "seawatertemperature"],
            "WETLABS_WQM": ["chlorophyll", "turbidity", "fluorescence"],
            "VIDEOCAM": ["imagery"],
            "CAMLIGHTS": ["imagery"],
            "PLANKTONSAMPLER": ["particlecount"],
            "ACOUSTICRECEIVER": ["detections"],
            "AISRECEIVER": ["aisdata"],
            "ICE_BUOY": ["icethickness", "airtemperature", "position"]
        }
        return device_properties.get(device_category, [])

    def get_device_deployments(self, device_code: str) -> List[Dict[str, Any]]:
        """
        Get deployment information for a specific device
        
        Args:
            device_code: Device code to get deployments for
            
        Returns:
            List of deployment information
        """
        params = {'deviceCode': device_code}
        response = self._make_request('deployments', params)
        
        if 'error' in response:
            logger.error(f"Failed to get deployments for {device_code}: {response}")
            return []
            
        deployments = response if isinstance(response, list) else []
        logger.info(f"Found {len(deployments)} deployments for device {device_code}")
        
        return deployments

    def close(self):
        """Close the session"""
        self.session.close()


def main():
    """Test the ONC API client"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ONC API Client')
    parser.add_argument('--location', default='CBYIP', help='Location code')
    parser.add_argument('--device', default='CTD', help='Device category')
    parser.add_argument('--property', default='seawatertemperature', help='Property code')
    parser.add_argument('--hours', type=int, default=24, help='Hours back to search')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = ONCAPIClient()
        
        print(f"Searching for {args.property} data from {args.device} devices at {args.location}")
        print(f"Looking back {args.hours} hours...")
        print("=" * 60)
        
        result = client.get_latest_data(
            location_code=args.location,
            device_category=args.device,
            property_code=args.property,
            hours_back=args.hours
        )
        
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        
        if result['status'] == 'success' and result['data']:
            formatted_data = client.format_sensor_data(result['data'])
            
            print(f"\nFound {len(formatted_data)} sensors:")
            print("-" * 60)
            
            for sensor in formatted_data:
                print(f"Sensor: {sensor['sensor_name']}")
                print(f"Property: {sensor['property_code']}")
                print(f"Latest: {sensor['latest_value']} {sensor['unit']}")
                print(f"Time: {sensor['latest_time']}")
                print(f"QA/QC: {sensor['qaqc_status']}")
                print(f"Total readings: {sensor['total_readings']}")
                print("-" * 60)
        
        print(f"\nMetadata:")
        print(json.dumps(result.get('metadata', {}), indent=2))
        
        client.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()