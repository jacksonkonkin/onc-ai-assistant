"""
Enhanced response formatter for ocean database queries.
Creates natural, conversational responses with educational context.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedResponseFormatter:
    """Creates natural, conversational responses for ocean database queries."""
    
    def __init__(self, llm_wrapper):
        """
        Initialize the enhanced response formatter.
        
        Args:
            llm_wrapper: LLM wrapper for generating natural language responses
        """
        self.llm_wrapper = llm_wrapper
        self.educational_context = self._load_educational_context()
    
    def format_enhanced_response(self, response: Dict[str, Any], 
                                conversation_context: str = "",
                                original_query: str = "") -> str:
        """
        Create an enhanced, conversational response for ocean data queries.
        
        Args:
            response: Database response dictionary
            conversation_context: Previous conversation context
            original_query: User's original question
            
        Returns:
            Enhanced natural language response
        """
        # Handle error and no data cases with natural language
        if response["status"] == "error":
            return self._format_error_response(response, original_query)
        
        if response["status"] == "no_data":
            return self._format_no_data_response(response, original_query)
        
        if response["status"] != "success" or not response["data"]:
            return f"I encountered an issue retrieving the data you requested. {response.get('message', '')}"
        
        # Format successful data response
        return self._format_success_response(response, conversation_context, original_query)
    
    def _format_success_response(self, response: Dict[str, Any], 
                                conversation_context: str, original_query: str) -> str:
        """Format a successful data response with natural language and educational context."""
        
        try:
            # Extract key information
            formatted_data = self._extract_formatted_data(response)
            if not formatted_data:
                # Provide helpful fallback message
                metadata = response.get("metadata", {})
                extracted_params = metadata.get('extracted_parameters', {})
                location = self._get_friendly_location_name(extracted_params.get('location_code', ''))
                property_name = self._get_friendly_property_name(extracted_params.get('property_code', ''))
                
                return f"""I found ocean data for {property_name.lower()} at {location}, but encountered a formatting issue while preparing the detailed response. 

The query was successfully processed and data was retrieved from the Ocean Networks Canada database. You might try asking about a different time period or parameter, or check back in a moment."""
            
            metadata = response.get("metadata", {})
            extracted_params = metadata.get('extracted_parameters', {})
            
            # Create the natural language response using LLM
            natural_response = self._generate_natural_response(
                formatted_data, extracted_params, original_query, conversation_context
            )
            
            # Add technical summary and API details
            technical_summary = self._create_technical_summary(response, formatted_data)
            
            return f"{natural_response}\n\n{technical_summary}"
            
        except Exception as e:
            logger.error(f"Error in _format_success_response: {e}")
            logger.error(f"Response structure: {type(response)} - {str(response)[:200]}...")
            
            # Try to provide a basic response using fallback
            try:
                metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
                extracted_params = metadata.get('extracted_parameters', {}) if isinstance(metadata, dict) else {}
                location = self._get_friendly_location_name(extracted_params.get('location_code', ''))
                property_name = self._get_friendly_property_name(extracted_params.get('property_code', ''))
                
                return f"""I successfully retrieved {property_name.lower()} data from {location}, but encountered a formatting issue while preparing the enhanced response.

The Ocean Networks Canada database query was completed successfully. The data shows current oceanographic conditions for the requested parameter. While I cannot provide the full enhanced explanation at this moment, the technical data retrieval was successful.

Please try your query again, or ask about a different oceanographic parameter."""
                
            except Exception as fallback_error:
                logger.error(f"Even fallback formatting failed: {fallback_error}")
                return f"I found the ocean data you requested, but encountered an unexpected error while formatting the response. The data retrieval was successful, but I'm having trouble presenting it properly. Please try your query again. Error details: {str(e)[:100]}..."
    
    def _generate_natural_response(self, formatted_data: List[Dict], 
                                  extracted_params: Dict, original_query: str,
                                  conversation_context: str) -> str:
        """Generate natural language response using LLM."""
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(formatted_data, extracted_params)
        educational_info = self._get_educational_context(extracted_params.get('property_code', ''))
        
        # Create prompt for natural response generation
        prompt = self._create_response_generation_prompt(
            original_query, data_summary, educational_info, conversation_context
        )
        
        try:
            natural_response = self.llm_wrapper.invoke(prompt)
            return natural_response.strip()
        except Exception as e:
            logger.error(f"Error generating natural response: {e}")
            # Fallback to structured response
            return self._create_fallback_response(formatted_data, extracted_params)
    
    def _create_response_generation_prompt(self, original_query: str, data_summary: str,
                                         educational_info: str, conversation_context: str) -> str:
        """Create prompt for generating natural language responses."""
        
        conversation_section = ""
        if conversation_context:
            conversation_section = f"""
CONVERSATION CONTEXT:
{conversation_context}

Note: Consider the previous conversation when crafting your response. If this is a follow-up question, reference the previous discussion appropriately.
"""
        
        return f"""You are an expert oceanographic data analyst responding to a user's question about ocean sensor data. 

Your task is to provide a natural, conversational response that:
1. Directly answers the user's question in a friendly, human-like way
2. Provides educational context about why this measurement matters
3. Explains what the data means in practical terms
4. Includes relevant oceanographic insights when appropriate

{conversation_section}

USER'S QUESTION: {original_query}

DATA RETRIEVED:
{data_summary}

EDUCATIONAL CONTEXT:
{educational_info}

INSTRUCTIONS:
- Start by directly answering the question in a conversational tone
- Explain the significance of the measurement and what it tells us
- Provide context about typical ranges, seasonal patterns, or scientific importance
- Make it educational but accessible to non-experts
- Keep the tone friendly and informative
- DO NOT include technical summaries, API calls, or raw data in this response
- Focus on interpretation and explanation rather than just reporting numbers

Generate a natural, educational response (2-4 paragraphs):"""
    
    def _prepare_data_summary(self, formatted_data: List[Dict], extracted_params: Dict) -> str:
        """Prepare a summary of the data for the LLM prompt."""
        if not formatted_data:
            return "No data available"
        
        summary_parts = []
        location = self._get_friendly_location_name(extracted_params.get('location_code', ''))
        property_name = self._get_friendly_property_name(extracted_params.get('property_code', ''))
        
        summary_parts.append(f"Location: {location}")
        summary_parts.append(f"Parameter: {property_name}")
        
        for i, sensor in enumerate(formatted_data, 1):
            value = sensor.get('latest_value', 'N/A')
            unit = sensor.get('unit', '')
            time = sensor.get('latest_time', '')
            sensor_name = sensor.get('sensor_name', f'Sensor {i}')
            
            # Format time nicely
            time_str = self._format_time_for_display(time)
            
            summary_parts.append(f"""
Sensor {i}: {sensor_name}
- Value: {value} {unit}
- Time: {time_str}
- Quality: {sensor.get('qaqc_status', 'Unknown')}
- Total readings: {sensor.get('total_readings', 'Unknown')}""")
        
        return "\n".join(summary_parts)
    
    def _get_educational_context(self, property_code: str) -> str:
        """Get educational context for the measured parameter."""
        return self.educational_context.get(property_code, 
            "This is an important oceanographic measurement that helps scientists understand marine conditions.")
    
    def _load_educational_context(self) -> Dict[str, str]:
        """Load educational context for different oceanographic parameters."""
        return {
            'seawatertemperature': """
            Sea water temperature is a fundamental oceanographic parameter that influences marine life, ocean circulation, and climate patterns. Temperature variations indicate seasonal changes, water mass movements, and can affect the solubility of gases like oxygen. In Arctic waters like Cambridge Bay, temperature monitoring is crucial for understanding ice formation, marine ecosystem health, and climate change impacts.
            """,
            
            'salinity': """
            Salinity measures the dissolved salt content in seawater and is critical for ocean density and circulation patterns. In Arctic regions, salinity is affected by freshwater input from ice melt, precipitation, and river runoff. Changes in salinity can indicate ice melt patterns, water mass mixing, and have important implications for marine organisms and ocean circulation.
            """,
            
            'pressure': """
            Water pressure measurements help determine depth and can indicate tidal variations, storm effects, and instrument positioning. Pressure data is essential for calculating accurate depth measurements and understanding underwater conditions that affect marine life and ocean processes.
            """,
            
            'dissolvedoxygen': """
            Dissolved oxygen is vital for marine life survival and indicates water quality and biological productivity. Low oxygen levels can signal pollution, eutrophication, or stratification effects. In Arctic waters, oxygen levels are influenced by temperature, ice cover, and biological activity.
            """,
            
            'ph': """
            Ocean pH measures acidity levels and is crucial for understanding ocean acidification, a major climate change impact. pH affects marine organisms, especially those with calcium carbonate shells or skeletons. Changes in pH can indicate environmental stress and ecosystem health.
            """,
            
            'turbidity': """
            Turbidity measures water clarity and indicates the amount of suspended particles in the water. High turbidity can result from sediment resuspension, runoff, or biological activity. It affects light penetration, photosynthesis, and marine habitat quality.
            """,
            
            'chlorophyll': """
            Chlorophyll concentration indicates phytoplankton abundance and primary productivity in marine ecosystems. It's a key indicator of marine food web health and nutrient availability. Seasonal chlorophyll patterns reveal ecosystem dynamics and environmental changes.
            """,
            
            'conductivity': """
            Conductivity measures water's ability to conduct electrical current and is directly related to salinity. It's used to calculate precise salinity values and understand water mass characteristics, mixing processes, and circulation patterns.
            """
        }
    
    def _format_time_for_display(self, time_str: str) -> str:
        """Format timestamp for user-friendly display."""
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.strftime('%B %d, %Y at %H:%M UTC')
        except:
            return time_str
    
    def _get_friendly_location_name(self, location_code: str) -> str:
        """Convert location code to friendly name."""
        location_names = {
            'CBYIP': 'Cambridge Bay (Ice Profiler)',
            'CBYSS': 'Cambridge Bay (Shallow Station)',
            'CBYIJ': 'Cambridge Bay (Ice Jam)',
            'CBYSP': 'Cambridge Bay (Shore Platform)',
            'CBYIU': 'Cambridge Bay (Ice Under)',
            'CBYDS': 'Cambridge Bay (Deep Station)'
        }
        return location_names.get(location_code, location_code or 'Unknown Location')
    
    def _get_friendly_property_name(self, property_code: str) -> str:
        """Convert property code to friendly name."""
        property_names = {
            'seawatertemperature': 'Sea Water Temperature',
            'salinity': 'Salinity',
            'pressure': 'Water Pressure',
            'dissolvedoxygen': 'Dissolved Oxygen',
            'ph': 'pH (Acidity)',
            'turbidity': 'Turbidity (Water Clarity)',
            'chlorophyll': 'Chlorophyll Concentration',
            'conductivity': 'Conductivity'
        }
        return property_names.get(property_code, property_code or 'Unknown Parameter')
    
    def _extract_formatted_data(self, response: Dict[str, Any]) -> List[Dict]:
        """Extract and format sensor data from response."""
        try:
            # Get raw data from response
            raw_data = response.get("data", [])
            logger.debug(f"Raw data type: {type(raw_data)}, length: {len(raw_data) if isinstance(raw_data, list) else 'N/A'}")
            
            if not raw_data:
                logger.warning("No data found in response")
                return []
            
            # Try to use the API client's format_sensor_data method
            from .onc_api_client import ONCAPIClient
            api_client = ONCAPIClient()
            formatted_data = api_client.format_sensor_data(raw_data)
            
            if formatted_data:
                logger.debug(f"API client formatted data successfully: {len(formatted_data)} sensors")
                return formatted_data
            else:
                logger.warning("API client formatting returned empty, using fallback")
                # Fallback: create basic formatted data from raw response
                return self._create_basic_formatted_data(raw_data, response)
                
        except Exception as e:
            logger.error(f"Error extracting formatted data: {e}")
            logger.error(f"Raw data structure: {str(raw_data)[:200] if 'raw_data' in locals() else 'raw_data not available'}...")
            
            # Fallback: try to create basic formatted data
            try:
                return self._create_basic_formatted_data(response.get("data", []), response)
            except Exception as fallback_error:
                logger.error(f"Fallback formatting also failed: {fallback_error}")
                return []
    
    def _create_basic_formatted_data(self, raw_data: List, response: Dict[str, Any]) -> List[Dict]:
        """Create basic formatted data structure when API client formatting fails."""
        try:
            formatted_sensors = []
            metadata = response.get("metadata", {})
            extracted_params = metadata.get("extracted_parameters", {})
            
            for i, sensor_data in enumerate(raw_data):
                # Handle both dict and other data structures
                if isinstance(sensor_data, dict):
                    sensor_name = sensor_data.get("sensorName", f"Sensor {i+1}")
                    sensor_code = sensor_data.get("sensorCode", "Unknown")
                    
                    # Extract values from data array if available
                    data_section = sensor_data.get("data", {})
                    values = data_section.get("values", []) if isinstance(data_section, dict) else []
                    times = data_section.get("times", []) if isinstance(data_section, dict) else []
                    
                    if values and times:
                        latest_value = values[-1] if values else "N/A"
                        latest_time = times[-1] if times else "Unknown"
                        unit = self._get_unit_for_property(extracted_params.get("property_code", ""))
                        
                        formatted_sensor = {
                            "sensor_name": sensor_name,
                            "sensor_code": sensor_code,
                            "latest_value": latest_value,
                            "latest_time": latest_time,
                            "unit": unit,
                            "qaqc_status": "Unknown",
                            "total_readings": len(values)
                        }
                        formatted_sensors.append(formatted_sensor)
            
            return formatted_sensors
            
        except Exception as e:
            logger.error(f"Error creating basic formatted data: {e}")
            return []
    
    def _get_unit_for_property(self, property_code: str) -> str:
        """Get appropriate unit for a property code."""
        units = {
            "seawatertemperature": "°C",
            "salinity": "PSU",
            "pressure": "dbar",
            "dissolvedoxygen": "ml/L",
            "ph": "pH units",
            "turbidity": "NTU",
            "chlorophyll": "mg/m³",
            "conductivity": "S/m"
        }
        return units.get(property_code, "")
    
    def _create_technical_summary(self, response: Dict[str, Any], formatted_data: List[Dict]) -> str:
        """Create technical summary section with data details and API information."""
        lines = ["─" * 80]
        lines.append("📊 TECHNICAL SUMMARY")
        lines.append("─" * 80)
        
        # Data summary
        lines.append("\n🔬 Sensor Data:")
        for i, sensor in enumerate(formatted_data, 1):
            lines.append(f"   • {sensor.get('sensor_name', f'Sensor {i}')}: "
                        f"{sensor.get('latest_value', 'N/A')} {sensor.get('unit', '')} "
                        f"({sensor.get('qaqc_status', 'Unknown')} quality)")
        
        # Query details
        metadata = response.get("metadata", {})
        extracted_params = metadata.get('extracted_parameters', {})
        
        lines.append(f"\n🎯 Query Details:")
        lines.append(f"   • Location: {extracted_params.get('location_code', 'Unknown')}")
        lines.append(f"   • Device: {extracted_params.get('device_category', 'Unknown')}")
        lines.append(f"   • Parameter: {extracted_params.get('property_code', 'Unknown')}")
        lines.append(f"   • Processing Time: {metadata.get('execution_time', 'Unknown')}s")
        
        # API calls information
        raw_api_responses = response.get("raw_api_responses", {})
        if raw_api_responses:
            lines.append(f"\n🔗 API Calls Made:")
            for endpoint, call_info in raw_api_responses.items():
                url = call_info.get('url', 'Unknown URL')
                lines.append(f"   • {endpoint}: {url}")
        
        return "\n".join(lines)
    
    def _create_fallback_response(self, formatted_data: List[Dict], extracted_params: Dict) -> str:
        """Create a fallback response when LLM generation fails."""
        if not formatted_data:
            return "I found the requested data, but encountered an issue formatting the response."
        
        sensor = formatted_data[0]
        value = sensor.get('latest_value', 'N/A')
        unit = sensor.get('unit', '')
        location = self._get_friendly_location_name(extracted_params.get('location_code', ''))
        property_name = self._get_friendly_property_name(extracted_params.get('property_code', ''))
        time_str = self._format_time_for_display(sensor.get('latest_time', ''))
        
        return f"""Based on the ocean sensor data, the {property_name.lower()} at {location} was {value} {unit} as of {time_str}.

This measurement provides valuable information about current ocean conditions in the area. {self._get_educational_context(extracted_params.get('property_code', '')).strip()}"""
    
    def _format_error_response(self, response: Dict[str, Any], original_query: str) -> str:
        """Format error responses in natural language."""
        error_message = response.get('message', 'Unknown error occurred')
        
        return f"""I apologize, but I encountered an issue while retrieving the ocean data you requested. 

The system reported: {error_message}

This could be due to a temporary connectivity issue with the Ocean Networks Canada database, or the specific parameters you requested might not be available. Please try rephrasing your question or asking about a different location or time period."""
    
    def _format_no_data_response(self, response: Dict[str, Any], original_query: str) -> str:
        """Format no data responses in natural language."""
        metadata = response.get("metadata", {})
        extracted_params = metadata.get('extracted_parameters', {})
        
        location = self._get_friendly_location_name(extracted_params.get('location_code', ''))
        property_name = self._get_friendly_property_name(extracted_params.get('property_code', ''))
        
        return f"""I searched the Ocean Networks Canada database but couldn't find {property_name.lower()} data for {location} during the requested time period.

This could mean:
• The sensors weren't collecting this type of data at that time
• There was a temporary gap in data collection
• The specific time period you asked about predates available data

You might try asking about a different time period, or I can suggest other types of oceanographic data available from this location. You can also ask about recent data, which is more likely to be available."""