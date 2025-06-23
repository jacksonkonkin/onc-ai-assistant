#!/usr/bin/env python3
"""
Enhanced Ocean Query Parameter Extractor
Maps natural language queries to specific ONC location/device/property codes
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from pathlib import Path

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
    
    def __init__(self):
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please add it to your .env file")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"
        
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

Return ONLY a JSON object with these exact fields:
{{
    "location_code": "exact ONC location code (e.g. CBYIP, CBYSS.M1)",
    "device_category": "exact ONC device category code (e.g. CTD, METSTN)", 
    "property_code": "exact ONC property code (e.g. seawatertemperature, windspeed)",
    "temporal_reference": "the exact date/time reference from query (use {current_year} for unspecified years)",
    "temporal_type": "single_date or date_range",
    "depth_meters": null or numeric depth if mentioned
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
        
        # Validate location code
        if location_code not in self.location_devices:
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
        
        # Build final result
        result = {
            "status": "success",
            "parameters": {
                "location_code": location_code,
                "device_category": device_category,
                "property_code": property_code,
                "start_time": start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z') if start_time else None,
                "end_time": end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z') if end_time else None,
                "depth_meters": depth
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
        """Convert natural language dates to datetime objects"""
        if not temporal_ref:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            return start_time, end_time
        
        now = datetime.now()
        temporal_ref = temporal_ref.lower().strip()
        
        # First try to parse as specific date formats
        date = self._parse_specific_date(temporal_ref, now)
        
        if date is None:
            # Handle relative dates
            if "today" in temporal_ref:
                date = now.date()
            elif "yesterday" in temporal_ref:
                date = (now - timedelta(days=1)).date()
            elif "last week" in temporal_ref:
                start_time = now - timedelta(weeks=1)
                end_time = now
                return start_time, end_time
            else:
                # Handle day names
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                day_found = False
                for i, day in enumerate(days):
                    if day in temporal_ref:
                        current_weekday = now.weekday()
                        days_diff = i - current_weekday
                        if days_diff >= 0:
                            days_diff -= 7  # Go to previous week
                        date = (now + timedelta(days=days_diff)).date()
                        day_found = True
                        break
                
                if not day_found:
                    # Default to yesterday to avoid future dates
                    date = (now - timedelta(days=1)).date()
        
        # For single dates, create time range (use a smaller range for better results)
        if temporal_type == "single_date":
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(hours=12)  # 12 hour window instead of 24
        else:
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=1)  # 1 day range instead of 7
        
        return start_time, end_time

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

    def get_available_options(self) -> Dict:
        """Return all available ONC codes for reference"""
        return {
            "locations": self.location_devices,
            "devices": self.device_properties,
            "aliases": {
                "parameters": self.parameter_aliases,
                "locations": self.location_aliases
            }
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