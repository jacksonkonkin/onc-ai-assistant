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
        
        devices = response if isinstance(response, list) else []
        
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