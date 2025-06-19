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
            self.extractor = EnhancedParameterExtractor()
            self.api_client = ONCAPIClient(onc_token)
            
            # Initialize enhanced response formatter if LLM wrapper is available
            self.enhanced_formatter = None
            if llm_wrapper:
                self.enhanced_formatter = EnhancedResponseFormatter(llm_wrapper)
                logger.info("Enhanced response formatting enabled")
            
            logger.info("Ocean Query System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise

    def process_query(self, query: str, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query and return ONC API data
        
        Args:
            query: Natural language query
            include_metadata: Whether to include processing metadata
            
        Returns:
            Complete response with data and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query}'")
        
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
                    "extraction_result": extraction_result,
                    "execution_time": time.time() - start_time
                } if include_metadata else None
            }
        
        params = extraction_result["parameters"]
        logger.info(f"Extracted parameters: {params}")
        
        # Step 2: Call ONC API with extracted parameters
        logger.info("Step 2: Calling ONC API...")
        
        try:
            api_result = self.api_client.search_data(
                location_code=params["location_code"],
                device_category=params["device_category"],
                property_code=params["property_code"],
                date_from=params["start_time"],
                date_to=params["end_time"],
                row_limit=100
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
        
        # Format successful response
        formatted_data = self.api_client.format_sensor_data(response["data"])
        
        if not formatted_data:
            return "INFO: No sensor data found in response"
        
        # Start with one-sentence summary
        lines = [self._format_summary_sentence(response, formatted_data)]
        lines.append("")
        
        # Add data details
        lines.append("DATA RETRIEVED:")
        lines.append("=" * 60)
        
        for i, sensor in enumerate(formatted_data, 1):
            lines.append(f"\n{i}. {sensor['sensor_name']}")
            lines.append(f"   Latest Value: {sensor['latest_value']} {sensor['unit']}")
            lines.append(f"   Time: {sensor['latest_time']}")
            lines.append(f"   QA/QC Status: {sensor['qaqc_status']}")
            lines.append(f"   Total Readings: {sensor['total_readings']}")
        
        # Add detailed query information
        lines.append("")
        lines.append(self._format_query_details(response))
        
        # Add API calls information
        lines.append("")
        lines.append(self._format_api_calls(response))
        
        # Add suggestions
        lines.append("")
        lines.append(self._format_suggestions(response))
        
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
            
            return f"RESULT: The latest {property_code} at {location} was {value} {unit} on {time_str}."
        
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
            "OCEAN NETWORKS CANADA API CALLS:",
            "=" * 60
        ]
        
        # Show devices API call
        if "devices_request" in raw_responses:
            devices_req = raw_responses["devices_request"]
            if "_debug_info" in devices_req:
                debug_info = devices_req["_debug_info"]
                params = debug_info.get('params', {})
                clean_params = {k: v for k, v in params.items() if k != 'token'}
                
                lines.extend([
                    "",
                    "1. Get Available Devices:",
                    f"   URL: {debug_info.get('url', 'Unknown')}",
                    f"   Parameters: {clean_params}",
                    f"   Response: Found {len(devices_req.get('data', []))} available devices"
                ])
        
        # Show sensor data API calls
        if "scalar_data_requests" in raw_responses:
            for i, req in enumerate(raw_responses["scalar_data_requests"], 2):
                response_data = req.get('response', {})
                sensor_data = response_data.get('sensorData', [])
                device_name = req.get('device_name', 'Unknown')
                
                lines.extend([
                    "",
                    f"{i}. Get Sensor Data - {device_name}:",
                ])
                
                if "_debug_info" in response_data:
                    debug_info = response_data["_debug_info"]
                    params = debug_info.get('params', {})
                    clean_params = {k: v for k, v in params.items() if k != 'token'}
                    
                    lines.extend([
                        f"   URL: {debug_info.get('url', 'Unknown')}",
                        f"   Parameters: {clean_params}",
                    ])
                
                if sensor_data:
                    lines.append(f"   Response: SUCCESS - Found {len(sensor_data)} sensors with data")
                    if response["status"] == "success":
                        break  # Stop showing after successful device
                else:
                    lines.append("   Response: NO DATA - No data found")
        
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