#!/usr/bin/env python3
"""
Enhanced Ocean Query Parameter Extractor
Maps natural language queries to specific ONC location/device/property codes
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    current_dir = Path(__file__).parent
    for path in [current_dir] + list(current_dir.parents):
        env_file = path / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip('\'"')
                        os.environ[key.strip()] = value
            return True
    return False

load_env_file()

try:
    from groq import Groq
except ImportError:
    print("Error: groq package not installed")
    print("Install it with: pip install groq")
    sys.exit(1)


class EnhancedParameterExtractor:
    """Extract parameters and map to exact ONC codes"""
    
    def __init__(self, onc_client=None):
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please add it to your .env file")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"
        
        # Store ONC API client for location discovery
        self.onc_client = onc_client
        
        # Load ONC codes mappings
        self._load_onc_codes()
        
        # Enhanced parameter mappings for natural language
        self.parameter_aliases = {
            "temp": "seawatertemperature",
            "temperature": "seawatertemperature", 
            "water temp": "seawatertemperature",
            "sea temperature": "seawatertemperature",
            "how hot": "seawatertemperature",
            "how cold": "seawatertemperature",
            "how warm": "seawatertemperature",
            "salt": "salinity",
            "saltiness": "salinity", 
            "salt content": "salinity",
            "o2": "oxygen",
            "dissolved oxygen": "oxygen",
            "oxygen content": "oxygen",
            "pressure": "pressure",
            "depth": "depth",
            "chlorophyll": "chlorophyll",
            "turbidity": "turbidityntu",
            "ph": "ph",
            "acidity": "ph",
            "conductivity": "conductivity",
            "air temp": "airtemperature",
            "air temperature": "airtemperature",
            "wind": "windspeed",
            "wind speed": "windspeed",
            "wind direction": "winddirection",
            "humidity": "relativehumidity",
            "pressure atmospheric": "absolutebarometricpressure",
            "barometric pressure": "absolutebarometricpressure",
            # Acoustic and ship noise mappings
            "ship noise": "soundpressurelevel",
            "acoustic": "soundpressurelevel",
            "underwater sound": "soundpressurelevel",
            "noise levels": "soundpressurelevel",
            "sound pressure": "soundpressurelevel",
            "ambient noise": "soundpressurelevel",
            "hydrophone data": "soundpressurelevel",
            "acoustic data": "soundpressurelevel",
            "sound": "soundpressurelevel",
            "underwater noise": "soundpressurelevel",
            "vessel noise": "soundpressurelevel",
            "marine noise": "soundpressurelevel"
        }
        
        # Location name mappings
        self.location_aliases = {
            "cambridge bay": "CBYIP",
            "iqaluktuuttiaq": "CBYIP", 
            "cambridge bay ice": "CBYSP",
            "cambridge bay shore": "CBYSS",
            "cambridge bay met 1": "CBYSS.M1",
            "cambridge bay met 2": "CBYSS.M2",
            "cambridge bay weather": "CBYSS.M1"
        }
        
        # Statistical keywords for detection
        self.statistical_keywords = {
            "aggregation": ["min", "minimum", "max", "maximum", "avg", "average", "mean", 
                          "sum", "total", "count", "median", "mode", "std", "stdev", 
                          "variance", "var", "range"],
            "comparison": ["higher", "lower", "greater", "less", "above", "below", 
                         "exceeds", "under", "compared to", "vs", "versus", "difference"],
            "temporal": ["trend", "change", "increase", "decrease", "rising", "falling",
                        "seasonal", "monthly", "daily", "weekly", "yearly", "over time"],
            "analysis": ["correlation", "relationship", "pattern", "analysis", "statistics",
                        "statistical", "distribution", "outlier", "anomaly"]
        }
        
        # Time window patterns for statistical analysis
        self.time_window_patterns = {
            "minute": ["minute", "min", "1min", "5min", "15min", "30min"],
            "hourly": ["hour", "hourly", "hr", "1hr", "2hr", "6hr", "12hr"],
            "daily": ["day", "daily", "per day", "each day", "1day", "24hr"],
            "weekly": ["week", "weekly", "per week", "each week", "7day"],
            "monthly": ["month", "monthly", "per month", "each month", "30day"],
            "yearly": ["year", "yearly", "annual", "per year", "each year", "365day"],
            "seasonal": ["season", "seasonal", "spring", "summer", "fall", "winter"]
        }

    def _load_onc_codes(self):
        """Load and parse ONC location/device/property codes"""
        codes_file = Path(__file__).parent / "location_device_property_codes_edited.txt"
        
        self.location_devices = {}
        self.device_properties = {}
        
        try:
            with open(codes_file, 'r') as f:
                content = f.read()
            
            # Parse location to device mappings
            location_section = content.split("*DEVICE CATEGORY CODE to PROPERTY CODES*")[0]
            current_location = None
            
            for line in location_section.split('\n'):
                line = line.strip()
                if not line or line.startswith('*'):
                    continue
                
                if not line.startswith('├──') and not line.startswith('└──'):
                    current_location = line
                    self.location_devices[current_location] = []
                elif line.startswith('├──') or line.startswith('└──'):
                    if current_location:
                        device = line.replace('├──', '').replace('└──', '').strip()
                        self.location_devices[current_location].append(device)
            
            # Parse device to property mappings
            property_section = content.split("*DEVICE CATEGORY CODE to PROPERTY CODES*")[1]
            current_device = None
            
            for line in property_section.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if not line.startswith('├──') and not line.startswith('└──'):
                    current_device = line
                    self.device_properties[current_device] = []
                elif line.startswith('├──') or line.startswith('└──'):
                    if current_device:
                        prop = line.replace('├──', '').replace('└──', '').strip()
                        if prop != "(No properties available)":
                            self.device_properties[current_device].append(prop)
                            
        except FileNotFoundError:
            print("Warning: ONC codes file not found, using defaults")
            self._setup_default_codes()

    def _setup_default_codes(self):
        """Setup default codes if file not found"""
        self.location_devices = {
            "CBYIP": ["CTD", "HYDROPHONE", "OXYSENSOR", "PHSENSOR"],
            "CBYSS.M1": ["METSTN"],
            "CBYSS.M2": ["METSTN"]
        }
        
        self.device_properties = {
            "CTD": ["seawatertemperature", "salinity", "pressure", "depth", "conductivity"],
            "OXYSENSOR": ["oxygen", "seawatertemperature"],
            "PHSENSOR": ["ph", "seawatertemperature"],
            "METSTN": ["airtemperature", "windspeed", "humidity", "absolutebarometricpressure"],
            "ICEPROFILER": ["soundpressurelevel", "seawatertemperature", "icedraft", "pingtime"],
            "HYDROPHONE": ["amperage", "batterycharge", "voltage", "internaltemperature"]
        }

    def extract_parameters(self, query: str) -> Dict:
        """Extract and map parameters to ONC codes"""
        
        # Create enhanced prompt with ONC codes context
        system_prompt = f"""You are an expert at extracting ocean/weather data parameters and mapping them to Ocean Networks Canada (ONC) codes.

Available locations: {list(self.location_devices.keys())}
Available devices per location: {json.dumps(self.location_devices, indent=2)}
Available properties per device: {json.dumps(self.device_properties, indent=2)}

Extract parameters and return ONLY valid JSON with exact ONC codes."""

        current_year = datetime.now().year
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        extraction_prompt = f"""Extract parameters from this query and map to exact ONC codes:
Query: "{query}"

IMPORTANT: Current date is {current_date}. When extracting dates:
- If no year is specified, ALWAYS assume {current_year}
- Only use different years if explicitly mentioned in the query
- For dates like "April 12" without year, extract as "{current_year}-04-12"

CRITICAL: For download_requested field - look for ANY download intent:
- "Can you download it?" → download_requested: true
- "download temperature data" → download_requested: true  
- "export data as CSV" → download_requested: true
- "save data to file" → download_requested: true
- "what is the temperature" → download_requested: false

Return ONLY a JSON object with these exact fields:
{{
    "location_code": "exact ONC location code (e.g. CBYIP, CBYSS.M1)",
    "device_category": "exact ONC device category code (e.g. CTD, METSTN)", 
    "property_code": "exact ONC property code (e.g. seawatertemperature, windspeed)",
    "temporal_reference": "the exact date/time reference from query (use {current_year} for unspecified years)",
    "temporal_type": "single_date or date_range",
    "depth_meters": null or numeric depth if mentioned,
    "download_requested": boolean indicating if user wants to download/export data to files
}}

Mapping rules:
- For temperature/temp/hot/cold/warm → map to "seawatertemperature" if water-related, "airtemperature" if air/weather
- For ship noise/acoustic/sound/hydrophone/underwater noise → map to "soundpressurelevel" property with "ICEPROFILER" device
- For Cambridge Bay standard queries → use "CBYIP" location with "CTD" device
- For weather/wind/air queries → use "CBYSS.M1" or "CBYSS.M2" location with "METSTN" device  
- For salt/salinity → use "salinity" property with CTD device
- For acoustic/sound pressure measurements → use "ICEPROFILER" device at "CBYIP" location
- Always use exact codes from the available options
- If location unclear, default to "CBYIP"
- If device unclear for property, pick the most appropriate device that has that property

Download detection rules (IMPORTANT - check carefully):
- Set "download_requested": true if query contains ANY of these keywords: download, export, save, CSV, file, "get data files", "retrieve files"
- Set "download_requested": true if user asks "Can you download it?", "download it", "export it", "save it as file", "save to file"
- Set "download_requested": true for "download [parameter] data", "export [parameter] data", "get CSV data"
- Set "download_requested": false ONLY for queries asking about data values, latest readings, or just showing/displaying data
- When in doubt about download intent, prefer "download_requested": true if ANY download-related words are present

Return ONLY the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                top_p=1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                raw_params = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    raw_params = json.loads(json_content)
                else:
                    return {"status": "error", "message": "Failed to parse LLM response"}
            
            # Validate and enhance extracted parameters
            return self._validate_and_enhance(raw_params, query)
            
        except Exception as e:
            return {"status": "error", "message": f"LLM extraction failed: {e}"}

    def _validate_and_enhance(self, raw_params: Dict, original_query: str) -> Dict:
        """Validate extracted parameters and enhance with fallbacks"""
        
        # Extract basic parameters
        location_code = raw_params.get("location_code", "")
        device_category = raw_params.get("device_category", "")
        property_code = raw_params.get("property_code", "")
        temporal_ref = raw_params.get("temporal_reference", "")
        temporal_type = raw_params.get("temporal_type", "single_date")
        depth = raw_params.get("depth_meters")
        
        # Validate location code - try dynamic discovery first if available
        if location_code not in self.location_devices:
            # Try dynamic location discovery for Cambridge Bay queries
            discovered_location = self._discover_location(original_query)
            if discovered_location:
                location_code = discovered_location
            else:
                # Try to map from aliases
                query_lower = original_query.lower()
                location_code = None
                for alias, code in self.location_aliases.items():
                    if alias in query_lower:
                        location_code = code
                        break
                
                if not location_code:
                    location_code = "CBYIP"  # Default location

        # Validate device category exists for this location
        available_devices = self.location_devices.get(location_code, [])
        
        # Try dynamic device discovery if device category is not found
        if device_category not in available_devices:
            discovered_device = self._discover_device(original_query)
            if discovered_device:
                device_category = discovered_device
            else:
                # First check if current device has the requested property
                current_device_properties = self.device_properties.get(device_category, [])
                
                # If device doesn't exist OR device doesn't have the property, find appropriate device
                if (device_category not in available_devices or 
                    property_code not in current_device_properties):
                    # Try to find appropriate device for the property
                    device_category = self._find_device_for_property(property_code, available_devices)

        # Validate property code exists for the selected device
        available_properties = self.device_properties.get(device_category, [])
        if property_code not in available_properties:
            # Try parameter aliases
            query_lower = original_query.lower()
            mapped_property = None
            for alias, prop in self.parameter_aliases.items():
                if alias in query_lower and prop in available_properties:
                    mapped_property = prop
                    break
            
            if mapped_property:
                property_code = mapped_property
            else:
                # Default to first available property
                if available_properties:
                    property_code = available_properties[0]

        # Parse temporal information
        start_time, end_time = self._parse_temporal_reference(temporal_ref, temporal_type)
        
        # Validate date range - check for future dates
        validation_errors = self._validate_date_range(start_time, end_time, original_query)
        if validation_errors:
            return {
                "status": "error",
                "message": validation_errors,
                "data": None
            }
        
        # Build final result
        result = {
            "status": "success",
            "parameters": {
                "location_code": location_code,
                "device_category": device_category,
                "property_code": property_code,
                "temporal_reference": temporal_ref if temporal_ref else None,
                "temporal_type": temporal_type,
                "start_time": start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z') if start_time else None,
                "end_time": end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z') if end_time else None,
                "depth_meters": depth,
                "download_requested": raw_params.get("download_requested", self._fallback_download_detection(original_query))
            },
            "metadata": {
                "original_query": original_query,
                "raw_extraction": raw_params,
                "available_devices": available_devices,
                "available_properties": available_properties,
                "model_used": self.model
            }
        }
        
        return result

    def _fallback_download_detection(self, query: str) -> bool:
        """
        Fallback method to detect download intent if LLM missed it
        """
        query_lower = query.lower()
        
        # Download keywords
        download_keywords = [
            'download', 'export', 'save', 'csv', 'file', 'files',
            'retrieve files', 'get data files', 'save to file',
            'export data', 'download data', 'save data'
        ]
        
        # Download phrases
        download_phrases = [
            'can you download', 'download it', 'export it', 
            'save it', 'get csv', 'as csv', 'to file'
        ]
        
        # Check for download keywords
        for keyword in download_keywords:
            if keyword in query_lower:
                return True
        
        # Check for download phrases
        for phrase in download_phrases:
            if phrase in query_lower:
                return True
        
        return False

    def _find_device_for_property(self, property_code: str, available_devices: List[str]) -> str:
        """Find appropriate device that has the requested property"""
        for device in available_devices:
            if property_code in self.device_properties.get(device, []):
                return device
        
        # Fallback logic for specific property types
        
        # Acoustic properties should use ICEPROFILER
        if "sound" in property_code.lower() or property_code == "soundpressurelevel":
            if "ICEPROFILER" in available_devices:
                return "ICEPROFILER"
            elif "HYDROPHONE" in available_devices:
                return "HYDROPHONE"  # Fallback, though may not have acoustic data
        
        # Temperature properties should use CTD or weather station
        if any(temp_term in property_code.lower() for temp_term in ["temp", "seawatertemperature"]):
            if "CTD" in available_devices:
                return "CTD"
            elif "METSTN" in available_devices:
                return "METSTN"
        
        # Return first available device
        return available_devices[0] if available_devices else "CTD"

    def _parse_temporal_reference(self, temporal_ref: str, temporal_type: str) -> Tuple[datetime, datetime]:
        """Convert natural language dates and times to datetime objects with enhanced interval support"""
        import re
        
        if not temporal_ref:
            # For no temporal reference, default to "latest" - very short window
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=5)  # 5-minute window for "latest"
            return start_time, end_time
        
        now = datetime.now()
        temporal_ref_lower = temporal_ref.lower().strip()
        
        # Handle date ranges FIRST (from X to Y) - including slash format
        # This must come before single date parsing to catch ranges properly
        if ' to ' in temporal_ref_lower or ' - ' in temporal_ref_lower or '/' in temporal_ref:
            return self._parse_date_range(temporal_ref, now)
        
        # Handle "week of" queries
        if 'week of' in temporal_ref_lower:
            # Extract the date after "week of"
            week_match = re.search(r'week of\s+(.+)', temporal_ref_lower)
            if week_match:
                date_str = week_match.group(1).strip()
                base_date = self._parse_specific_date(date_str, now)
                if base_date:
                    # Calculate start of week (Monday) and end of week (Sunday)
                    days_since_monday = base_date.weekday()
                    week_start = base_date - timedelta(days=days_since_monday)
                    week_end = week_start + timedelta(days=6)
                    
                    start_time = datetime.combine(week_start, datetime.min.time())
                    end_time = datetime.combine(week_end, datetime.max.time())
                    return start_time, end_time
        
        # Handle explicit intervals
        interval_patterns = {
            # Multi-unit intervals
            r'last (\d+) hours?': lambda m: (now - timedelta(hours=int(m.group(1))), now),
            r'past (\d+) hours?': lambda m: (now - timedelta(hours=int(m.group(1))), now),
            r'last (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
            r'past (\d+) days?': lambda m: (now - timedelta(days=int(m.group(1))), now),
            r'last (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
            r'past (\d+) weeks?': lambda m: (now - timedelta(weeks=int(m.group(1))), now),
            r'last (\d+) months?': lambda m: (now - timedelta(days=int(m.group(1))*30), now),
            r'past (\d+) months?': lambda m: (now - timedelta(days=int(m.group(1))*30), now),
            
            # Common intervals
            r'last hour': lambda m: (now - timedelta(hours=1), now),
            r'past hour': lambda m: (now - timedelta(hours=1), now),
            r'last 24 hours?': lambda m: (now - timedelta(hours=24), now),
            r'past 24 hours?': lambda m: (now - timedelta(hours=24), now),
            r'last week': lambda m: (now - timedelta(weeks=1), now),
            r'past week': lambda m: (now - timedelta(weeks=1), now),
            r'last month': lambda m: (now - timedelta(days=30), now),
            r'past month': lambda m: (now - timedelta(days=30), now),
            
            # Specific timeframes
            r'this week': lambda m: (now - timedelta(days=now.weekday()), now),
            r'this month': lambda m: (now.replace(day=1, hour=0, minute=0, second=0), now),
            
            # "Latest" and "current" - minimal intervals
            r'latest|current|now': lambda m: (now - timedelta(minutes=5), now),
        }
        
        # Check for interval patterns
        import re
        for pattern, handler in interval_patterns.items():
            match = re.search(pattern, temporal_ref_lower)
            if match:
                return handler(match)
        
        # Try to parse as specific date formats
        date = self._parse_specific_date(temporal_ref, now)
        
        if date is None:
            # Handle relative dates
            if "today" in temporal_ref_lower:
                date = now.date()
            elif "yesterday" in temporal_ref_lower:
                date = (now - timedelta(days=1)).date()
            else:
                # Handle day names
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                day_found = False
                for i, day in enumerate(days):
                    if day in temporal_ref_lower:
                        current_weekday = now.weekday()
                        days_diff = i - current_weekday
                        if days_diff >= 0:
                            days_diff -= 7  # Go to previous week
                        date = (now + timedelta(days=days_diff)).date()
                        day_found = True
                        break
                
                if not day_found:
                    # Default to latest 5-minute window for unknown temporal references
                    return now - timedelta(minutes=5), now
        
        # Parse specific time if mentioned
        specific_time = self._parse_specific_time(temporal_ref)
        
        # Determine interval scope based on context
        if temporal_type == "single_date" or "instant" in temporal_type:
            if specific_time is not None:
                # Use exact timestamp for precise time queries - small window
                start_time = datetime.combine(date, specific_time)
                end_time = start_time + timedelta(minutes=1)
            else:
                # For date-only queries, use the full day
                start_time = datetime.combine(date, datetime.min.time())
                end_time = start_time + timedelta(days=1) - timedelta(seconds=1)
        else:
            # Default to full day range
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=1) - timedelta(seconds=1)
        
        return start_time, end_time
    
    def _parse_date_range(self, temporal_ref: str, now: datetime) -> Tuple[datetime, datetime]:
        """Parse date ranges like '2024-01-01 to 2024-01-31', 'Monday to Friday', or '2024-01-01/2024-01-31'"""
        import re
        
        # Handle different range separators more carefully
        start_str = None
        end_str = None
        
        # Try different range separators in order of preference
        if ' to ' in temporal_ref:
            parts = temporal_ref.split(' to ')
            if len(parts) == 2:
                start_str, end_str = parts[0].strip(), parts[1].strip()
        elif '/' in temporal_ref and temporal_ref.count('/') == 1:
            # Only split on single forward slash (not part of date like 2024/07/20)
            # Check if it's a range separator by seeing if we have two ISO-style dates
            slash_pos = temporal_ref.find('/')
            if slash_pos > 0:
                potential_start = temporal_ref[:slash_pos].strip()
                potential_end = temporal_ref[slash_pos+1:].strip()
                # Check if both parts look like dates (YYYY-MM-DD format)
                if (re.match(r'\d{4}-\d{1,2}-\d{1,2}', potential_start) and 
                    re.match(r'\d{4}-\d{1,2}-\d{1,2}', potential_end)):
                    start_str, end_str = potential_start, potential_end
        elif ' - ' in temporal_ref:
            # Only split on spaced dashes (not date dashes)
            parts = temporal_ref.split(' - ')
            if len(parts) == 2:
                start_str, end_str = parts[0].strip(), parts[1].strip()
        
        if not start_str or not end_str:
            # Fallback to last 24 hours
            return now - timedelta(days=1), now
        
        # Try to parse both parts as dates
        start_date = self._parse_specific_date(start_str, now)
        end_date = self._parse_specific_date(end_str, now)
        
        if start_date and end_date:
            start_time = datetime.combine(start_date, datetime.min.time())
            end_time = datetime.combine(end_date, datetime.max.time())
            return start_time, end_time
        
        # Fallback
        return now - timedelta(days=1), now

    def _validate_date_range(self, start_time: datetime, end_time: datetime, query: str) -> Optional[str]:
        """Validate date range and return error message if invalid"""
        if not start_time or not end_time:
            return None
        
        now = datetime.now()
        
        # Make sure we're comparing timezone-aware or timezone-naive consistently
        if start_time.tzinfo is None:
            comparison_start = start_time
            comparison_now = now
        else:
            comparison_start = start_time.replace(tzinfo=None)
            comparison_now = now
        
        # Check for future dates (allowing some tolerance for time zones and processing delays)
        if comparison_start > comparison_now:
            days_in_future = (comparison_start - comparison_now).days
            if days_in_future >= 0:
                return f"I cannot provide data for future dates. The requested date ({comparison_start.strftime('%B %d, %Y')}) is {days_in_future} days in the future. Ocean sensor data is only available for past dates. Please ask for a date in the past."
        
        # Also check if the date is very recent (data may not be processed yet)
        days_ago = (comparison_now - comparison_start).days
        if days_ago < 1:  # Less than 1 day ago
            return f"The requested date ({comparison_start.strftime('%B %d, %Y')}) is very recent. Ocean sensor data typically has a 1-2 day processing delay. Please try requesting data from at least 2 days ago for more reliable results."
        
        # Check for very old dates (before ONC data availability)
        onc_start_date = datetime(2007, 1, 1)  # ONC started operations around 2007
        if end_time < onc_start_date:
            return f"The requested date ({start_time.strftime('%B %d, %Y')}) is before Ocean Networks Canada data collection began. Please request data from 2007 onwards."
        
        # Check for excessively long date ranges
        duration = end_time - start_time
        if duration.days > 365:  # More than a year
            return f"The requested date range spans {duration.days} days, which is quite large. For better performance, please try a shorter date range (up to 1 year)."
        
        return None

    def _parse_specific_time(self, temporal_ref: str) -> Optional:
        """
        Parse specific time expressions like '4:00pm', '16:00', 'at 4pm', etc.
        Returns a time object if parsing succeeds, None otherwise.
        """
        import re
        from datetime import time
        
        # Common time patterns
        time_patterns = [
            # 12-hour format with AM/PM
            r'(\d{1,2}):(\d{2})\s*(am|pm)',  # 4:00pm, 12:30am
            r'(\d{1,2})\s*(am|pm)',          # 4pm, 12am
            r'(\d{1,2}):(\d{2})',            # 16:00, 4:00 (24-hour assumed if > 12)
            r'at\s+(\d{1,2}):(\d{2})\s*(am|pm)',  # at 4:00pm
            r'at\s+(\d{1,2})\s*(am|pm)',          # at 4pm
            r'at\s+(\d{1,2}):(\d{2})',            # at 16:00
        ]
        
        temporal_ref = temporal_ref.lower().strip()
        
        for pattern in time_patterns:
            match = re.search(pattern, temporal_ref)
            if match:
                groups = match.groups()
                
                if len(groups) == 3:  # Hour, minute, AM/PM
                    hour, minute, am_pm = groups
                    hour, minute = int(hour), int(minute)
                    
                    # Convert to 24-hour format
                    if am_pm == 'pm' and hour != 12:
                        hour += 12
                    elif am_pm == 'am' and hour == 12:
                        hour = 0
                        
                elif len(groups) == 2:
                    if groups[1] in ['am', 'pm']:  # Hour only with AM/PM
                        hour, am_pm = groups
                        hour, minute = int(hour), 0
                        
                        # Convert to 24-hour format
                        if am_pm == 'pm' and hour != 12:
                            hour += 12
                        elif am_pm == 'am' and hour == 12:
                            hour = 0
                    else:  # Hour and minute (24-hour format)
                        hour, minute = int(groups[0]), int(groups[1])
                
                # Validate time
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return time(hour, minute)
        
        return None

    def _parse_specific_date(self, temporal_ref: str, now: datetime) -> Optional:
        """
        Parse specific date formats like ISO dates, month/day combinations, etc.
        Returns a date object if parsing succeeds, None otherwise
        """
        from datetime import datetime as dt
        import re
        
        # Try ISO date format (YYYY-MM-DD)
        iso_match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})', temporal_ref)
        if iso_match:
            try:
                year, month, day = map(int, iso_match.groups())
                return dt(year, month, day).date()
            except ValueError:
                pass
        
        # Try to parse month names with days (e.g., "april 12", "december 25")
        months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # Look for month name + day patterns
        for month_name, month_num in months.items():
            # Pattern: "april 12", "april 12th", "12 april", "12th april"
            patterns = [
                rf'{month_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?',  # "april 12th"
                rf'(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_name}'   # "12th april"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, temporal_ref, re.IGNORECASE)
                if match:
                    try:
                        day = int(match.group(1))
                        # Determine which year to use
                        current_year = now.year
                        
                        # Try current year first
                        try:
                            candidate_date = dt(current_year, month_num, day).date()
                            
                            # If the date is more than 6 months in the future, assume previous year
                            # If the date is more than 6 months in the past, could be next year
                            days_diff = (candidate_date - now.date()).days
                            
                            if days_diff > 180:  # More than 6 months in future
                                candidate_date = dt(current_year - 1, month_num, day).date()
                            elif days_diff < -180:  # More than 6 months in past
                                candidate_date = dt(current_year + 1, month_num, day).date()
                            
                            return candidate_date
                            
                        except ValueError:
                            # Invalid date (e.g., Feb 30)
                            continue
                            
                    except (ValueError, IndexError):
                        continue
        
        # Try MM/DD or DD/MM patterns
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY or DD/MM/YYYY  
            r'(\d{1,2})/(\d{1,2})',          # MM/DD or DD/MM
            r'(\d{1,2})-(\d{1,2})-(\d{4})',  # MM-DD-YYYY or DD-MM-YYYY
            r'(\d{1,2})-(\d{1,2})'           # MM-DD or DD-MM
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, temporal_ref)
            if match:
                try:
                    if len(match.groups()) == 3:  # Has year
                        part1, part2, year = map(int, match.groups())
                    else:  # No year, use current year
                        part1, part2 = map(int, match.groups())
                        year = now.year
                    
                    # Try both MM/DD and DD/MM interpretations
                    for month, day in [(part1, part2), (part2, part1)]:
                        try:
                            if 1 <= month <= 12 and 1 <= day <= 31:
                                candidate_date = dt(year, month, day).date()
                                
                                # Apply same year adjustment logic as above
                                if len(match.groups()) == 2:  # No year specified
                                    days_diff = (candidate_date - now.date()).days
                                    if days_diff > 180:
                                        candidate_date = dt(year - 1, month, day).date()
                                    elif days_diff < -180:
                                        candidate_date = dt(year + 1, month, day).date()
                                
                                return candidate_date
                        except ValueError:
                            continue
                            
                except (ValueError, IndexError):
                    continue
        
        return None

    def _discover_location(self, query: str) -> Optional[str]:
        """
        Discover Cambridge Bay location using ONC API client
        
        Args:
            query: User query that may contain location references
            
        Returns:
            Location code if found, None otherwise
        """
        if not self.onc_client or not query:
            return None
        
        try:
            query_lower = query.lower()
            
            # Check if query mentions Cambridge Bay
            if any(term in query_lower for term in ["cambridge bay", "cambridge", "iqaluktuuttiaq"]):
                # Get Cambridge Bay locations from ONC API
                cambridge_locations = self.onc_client.find_cambridge_bay_locations()
                
                if cambridge_locations:
                    # For queries about locations/deployments, return the main CBYIP location
                    # For specific device/sensor queries, could return more specific locations
                    if any(term in query_lower for term in ["location", "where", "site", "station"]):
                        # Return first location for general location queries
                        return cambridge_locations[0].get('locationCode', 'CBYIP')
                    else:
                        # For data queries, use CBYIP as default Cambridge Bay location
                        return 'CBYIP'
                        
        except Exception as e:
            # If API call fails, fall back to default
            logger.warning(f"Location discovery failed: {e}")
            return None
        
        return None

    def _discover_device(self, query: str) -> Optional[str]:
        """
        Discover Cambridge Bay devices using ONC API client
        
        Args:
            query: User query that may contain device references
            
        Returns:
            Device category code if found, None otherwise
        """
        if not self.onc_client or not query:
            return None
        
        try:
            query_lower = query.lower()
            
            # Check for device discovery queries
            device_discovery_terms = [
                "device", "sensor", "instrument", "equipment", "ctd", "hydrophone",
                "weather station", "oxygen sensor", "ph sensor", "ice profiler",
                "what sensors", "what devices", "what instruments"
            ]
            
            if any(term in query_lower for term in device_discovery_terms):
                # Extract device type from query
                device_type = self._extract_device_type_from_query(query_lower)
                
                if device_type:
                    # Try to find devices of this type at Cambridge Bay
                    devices = self.onc_client.discover_devices_by_type(device_type)
                    
                    if devices:
                        # Return the device category of the first matching device
                        return devices[0].get('deviceCategoryCode')
                        
        except Exception as e:
            logger.warning(f"Device discovery failed: {e}")
            return None
        
        return None

    def _extract_device_type_from_query(self, query_lower: str) -> Optional[str]:
        """
        Extract device type from natural language query
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            Device type if found, None otherwise
        """
        # Device type patterns to look for
        device_patterns = {
            'ctd': ['ctd', 'conductivity', 'temperature', 'depth'],
            'hydrophone': ['hydrophone', 'acoustic', 'underwater sound', 'sound pressure'],
            'weather': ['weather', 'meteorological', 'wind', 'air temperature', 'met station'],
            'oxygen': ['oxygen', 'o2', 'dissolved oxygen'],
            'ph': ['ph', 'acidity', 'ph sensor'],
            'ice': ['ice', 'ice profiler', 'ice draft']
        }
        
        for device_type, patterns in device_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return device_type
        
        # If no specific pattern matched, try generic terms
        if any(term in query_lower for term in ['sensor', 'device', 'instrument']):
            # Look for measurement types that might indicate device type
            if any(term in query_lower for term in ['temperature', 'salinity', 'conductivity']):
                return 'ctd'
            elif any(term in query_lower for term in ['wind', 'air', 'weather']):
                return 'weather'
            elif any(term in query_lower for term in ['sound', 'acoustic', 'noise']):
                return 'hydrophone'
        
        return None

    def detect_statistical_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect if a query has statistical intent and extract statistical parameters
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with statistical intent detection results
        """
        query_lower = query.lower()
        
        # Initialize detection results
        statistical_intent = {
            'is_statistical': False,
            'operations': [],
            'time_aggregation': None,
            'comparison_type': None,
            'analysis_type': None,
            'confidence': 0.0
        }
        
        confidence_score = 0.0
        
        # Check for aggregation operations
        found_operations = []
        for operation in self.statistical_keywords['aggregation']:
            if operation in query_lower:
                found_operations.append(operation)
                confidence_score += 1.0
        
        if found_operations:
            statistical_intent['operations'] = found_operations
            statistical_intent['is_statistical'] = True
        
        # Check for comparison keywords
        comparison_found = []
        for comparison in self.statistical_keywords['comparison']:
            if comparison in query_lower:
                comparison_found.append(comparison)
                confidence_score += 0.8
        
        if comparison_found:
            statistical_intent['comparison_type'] = comparison_found[0]
            statistical_intent['is_statistical'] = True
        
        # Check for temporal analysis
        temporal_found = []
        for temporal in self.statistical_keywords['temporal']:
            if temporal in query_lower:
                temporal_found.append(temporal)
                confidence_score += 0.9
        
        if temporal_found:
            statistical_intent['analysis_type'] = 'temporal'
            statistical_intent['is_statistical'] = True
        
        # Check for general analysis keywords
        analysis_found = []
        for analysis in self.statistical_keywords['analysis']:
            if analysis in query_lower:
                analysis_found.append(analysis)
                confidence_score += 0.7
        
        if analysis_found:
            if not statistical_intent['analysis_type']:
                statistical_intent['analysis_type'] = 'general'
            statistical_intent['is_statistical'] = True
        
        # Detect time aggregation windows
        for window_type, patterns in self.time_window_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    statistical_intent['time_aggregation'] = window_type
                    confidence_score += 0.6
                    statistical_intent['is_statistical'] = True
                    break
            if statistical_intent['time_aggregation']:
                break
        
        # Calculate final confidence score
        max_possible_score = 4.3  # Rough estimate of max score
        statistical_intent['confidence'] = min(confidence_score / max_possible_score, 1.0)
        
        # If we found any statistical indicators, mark as statistical
        if confidence_score > 0:
            statistical_intent['is_statistical'] = True
        
        return statistical_intent
    
    def extract_statistical_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract statistical parameters from a query
        
        Args:
            query: Natural language query with statistical intent
            
        Returns:
            Dictionary with extracted statistical parameters
        """
        # First get basic parameters
        basic_params = self.extract_parameters(query)
        
        # Then get statistical intent
        statistical_intent = self.detect_statistical_intent(query)
        
        # Enhanced statistical parameter extraction using LLM
        if statistical_intent['is_statistical']:
            try:
                enhanced_stats = self._extract_statistical_parameters_with_llm(query, statistical_intent)
                
                # Merge basic parameters with statistical parameters
                result = {
                    'status': basic_params.get('status', 'success'),
                    'parameters': basic_params.get('parameters', {}),
                    'statistical_intent': statistical_intent,
                    'statistical_parameters': enhanced_stats,
                    'is_statistical_query': True
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error in statistical parameter extraction: {e}")
                # Fall back to basic detection
                result = {
                    'status': basic_params.get('status', 'success'),
                    'parameters': basic_params.get('parameters', {}),
                    'statistical_intent': statistical_intent,
                    'statistical_parameters': self._extract_basic_statistical_params(query),
                    'is_statistical_query': True
                }
                return result
        else:
            # Not a statistical query, return basic parameters
            result = basic_params.copy()
            result['is_statistical_query'] = False
            result['statistical_intent'] = statistical_intent
            return result
    
    def _extract_statistical_parameters_with_llm(self, query: str, statistical_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to extract detailed statistical parameters
        
        Args:
            query: User query
            statistical_intent: Basic statistical intent detection results
            
        Returns:
            Detailed statistical parameters
        """
        system_prompt = """You are an expert at extracting statistical analysis parameters from natural language queries about oceanographic data.

Extract statistical parameters and return ONLY valid JSON with these fields:
- operations: list of statistical operations (min, max, avg, sum, count, etc.)
- time_window: time aggregation window (hourly, daily, weekly, monthly, yearly, etc.)
- comparison_criteria: what to compare (locations, time periods, thresholds)
- grouping: how to group data (by time, location, device, depth)
- threshold_values: any numeric thresholds mentioned
- analysis_type: type of analysis (basic_stats, trend, correlation, seasonal)

Available operations: min, max, avg, mean, sum, count, median, std, variance, range, trend, correlation"""

        extraction_prompt = f"""Extract statistical parameters from this query:
Query: "{query}"

Pre-detected intent: {json.dumps(statistical_intent, indent=2)}

Return ONLY valid JSON with the statistical parameters."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Clean and parse JSON
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            elif response_text.startswith('```'):
                response_text = response_text.replace('```', '').strip()
            
            statistical_params = json.loads(response_text)
            
            # Validate and clean the parameters
            validated_params = self._validate_statistical_parameters(statistical_params)
            
            return validated_params
            
        except Exception as e:
            logger.error(f"LLM statistical parameter extraction failed: {e}")
            return self._extract_basic_statistical_params(query)
    
    def _extract_basic_statistical_params(self, query: str) -> Dict[str, Any]:
        """
        Extract basic statistical parameters using pattern matching
        
        Args:
            query: User query
            
        Returns:
            Basic statistical parameters
        """
        query_lower = query.lower()
        
        # Extract operations
        operations = []
        for operation in self.statistical_keywords['aggregation']:
            if operation in query_lower:
                operations.append(operation)
        
        # Extract time window  
        time_window = None
        for window_type, patterns in self.time_window_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    time_window = window_type
                    break
            if time_window:
                break
        
        # Extract comparison criteria
        comparison_criteria = []
        for comparison in self.statistical_keywords['comparison']:
            if comparison in query_lower:
                comparison_criteria.append(comparison)
        
        # Determine analysis type
        analysis_type = 'basic_stats'
        if any(word in query_lower for word in ['trend', 'change', 'increase', 'decrease']):
            analysis_type = 'trend'
        elif any(word in query_lower for word in ['correlation', 'relationship']):
            analysis_type = 'correlation'
        elif any(word in query_lower for word in ['seasonal', 'monthly', 'yearly']):
            analysis_type = 'seasonal'
        
        return {
            'operations': operations if operations else ['avg', 'min', 'max'],
            'time_window': time_window,
            'comparison_criteria': comparison_criteria,
            'grouping': self._extract_grouping_from_query(query_lower),
            'threshold_values': self._extract_numeric_thresholds(query),
            'analysis_type': analysis_type
        }
    
    def _extract_grouping_from_query(self, query_lower: str) -> Dict[str, bool]:
        """Extract grouping criteria from query"""
        return {
            'by_time': any(phrase in query_lower for phrase in ['by hour', 'by day', 'by month', 'hourly', 'daily', 'monthly']),
            'by_location': 'by location' in query_lower or 'per location' in query_lower,
            'by_device': 'by device' in query_lower or 'per device' in query_lower,
            'by_depth': 'by depth' in query_lower or 'per depth' in query_lower
        }
    
    def _extract_numeric_thresholds(self, query: str) -> List[float]:
        """Extract numeric thresholds from query"""
        import re
        
        # Find numbers in the query
        numbers = re.findall(r'-?\d+\.?\d*', query)
        
        # Convert to float and filter reasonable values for oceanographic data
        thresholds = []
        for num_str in numbers:
            try:
                num = float(num_str)
                # Filter reasonable oceanographic values (rough ranges)
                if -50 <= num <= 100:  # Temperature, salinity, etc.
                    thresholds.append(num)
            except ValueError:
                continue
        
        return thresholds
    
    def _validate_statistical_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean statistical parameters from LLM
        
        Args:
            params: Raw parameters from LLM
            
        Returns:
            Validated and cleaned parameters
        """
        validated = {
            'operations': [],
            'time_window': None,
            'comparison_criteria': [],
            'grouping': {},
            'threshold_values': [],
            'analysis_type': 'basic_stats'
        }
        
        # Validate operations
        if 'operations' in params and isinstance(params['operations'], list):
            valid_operations = []
            all_operations = self.statistical_keywords['aggregation']
            for op in params['operations']:
                if op.lower() in [o.lower() for o in all_operations]:
                    valid_operations.append(op.lower())
            validated['operations'] = valid_operations if valid_operations else ['avg', 'min', 'max']
        else:
            validated['operations'] = ['avg', 'min', 'max']
        
        # Validate time window
        if 'time_window' in params:
            time_window = params['time_window']
            if time_window and time_window.lower() in self.time_window_patterns:
                validated['time_window'] = time_window.lower()
        
        # Validate other fields
        validated['comparison_criteria'] = params.get('comparison_criteria', [])
        validated['grouping'] = params.get('grouping', {})
        validated['threshold_values'] = params.get('threshold_values', [])
        validated['analysis_type'] = params.get('analysis_type', 'basic_stats')
        
        return validated

    def get_available_options(self) -> Dict:
        """Return all available ONC codes for reference"""
        return {
            "locations": self.location_devices,
            "devices": self.device_properties,
            "aliases": {
                "parameters": self.parameter_aliases,
                "locations": self.location_aliases
            },
            "statistical_operations": self.statistical_keywords,
            "time_windows": self.time_window_patterns
        }


def main():
    """Test the enhanced parameter extractor"""
    try:
        extractor = EnhancedParameterExtractor()
        
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            result = extractor.extract_parameters(query)
            print(json.dumps(result, indent=2, default=str))
        else:
            # Interactive mode
            print("Enhanced Ocean Query Parameter Extractor")
            print("=" * 50)
            print("Example queries:")
            print("  - What is the temperature in Cambridge Bay today?")
            print("  - Show me wind speed at the weather station")
            print("  - Cambridge Bay salinity data from yesterday")
            print("\nType 'quit' to exit\n")
            
            while True:
                try:
                    query = input("Enter query: ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break
                    if not query:
                        continue
                    
                    result = extractor.extract_parameters(query)
                    print("\n" + "="*50)
                    print(json.dumps(result, indent=2, default=str))
                    print("="*50 + "\n")
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()