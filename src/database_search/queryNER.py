#!/usr/bin/env python3
"""
Ocean Query Parameter Extractor using Groq API with Llama 3 70B
Ready to use - just ensure GROQ_API_KEY is set in your environment
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    # Check current directory and parent directories for .env file
    current_dir = Path(__file__).parent
    for path in [current_dir] + list(current_dir.parents):
        env_file = path / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('\'"')
                        os.environ[key.strip()] = value
            return True
    return False

# Load .env file if it exists
load_env_file()

# Make sure groq is installed
try:
    from groq import Groq
except ImportError:
    print("Error: groq package not installed")
    print("Install it with: pip install groq")
    sys.exit(1)


class OceanQueryExtractorGroq:
    """Extract parameters from ocean data queries using Groq's Llama 3 70B"""
    
    def __init__(self):
        # Initialize Groq client with API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Please add it to your .env file: GROQ_API_KEY=your-api-key-here")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"
        
        # Conversation memory for partial queries
        self.pending_clarification = None
        
        # System and extraction prompts
        self.system_prompt = """You are an expert at extracting structured information from ocean data queries.
You must extract parameters and return ONLY valid JSON with no additional text or explanation."""
        
        self.extraction_prompt = """Extract parameters from this ocean data query:
Query: "{query}"

Return ONLY a JSON object with these exact fields:
{{
    "parameter": "measurement type (use exactly one of: temperature/salinity/pressure/oxygen/chlorophyll/turbidity/ph/conductivity)",
    "location": "location name exactly as mentioned in the query",
    "temporal_reference": "the exact date/time reference from the query",
    "temporal_type": "single_date or date_range",
    "depth_meters": null or the numeric depth value if mentioned
}}

Rules:
- For "how cold/warm/hot" → use "temperature"
- For "salt content/saltiness" → use "salinity"
- For "water temp" → use "temperature"
- Keep location names exactly as stated
- Keep temporal references exactly as stated
- Only include depth if explicitly mentioned as a number with units
- If a field is not mentioned, use null
- Return ONLY the JSON object, no other text"""
        
        # Parameter mappings
        self.parameter_aliases = {
            "temp": "temperature",
            "water temp": "temperature",
            "how hot": "temperature",
            "how cold": "temperature",
            "how warm": "temperature",
            "salt": "salinity",
            "saltiness": "salinity",
            "salt content": "salinity",
            "o2": "oxygen",
            "dissolved oxygen": "oxygen"
        }
        
        # Known locations
        self.locations = {
            "cambridge bay": "Cambridge Bay",
            "iqaluktuuttiaq": "Cambridge Bay",
            "saanich inlet": "Saanich Inlet",
            "strait of georgia": "Strait of Georgia"
        }
    
    def extract_with_llm(self, query: str) -> Dict:
        """Call Groq API to extract parameters"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.extraction_prompt.format(query=query)}
                ],
                temperature=0.1,
                max_tokens=300,
                top_p=1,
                stream=False
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse the response as JSON
            try:
                # First try direct parsing
                return json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to find JSON in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_content = content[json_start:json_end]
                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        return self.fix_and_parse_json(json_content, content)
                else:
                    print(f"Warning: Could not find JSON structure in response: {content}")
                    return self.fix_and_parse_json(content, content)
                    
        except Exception as e:
            print(f"Groq API error: {e}")
            return {}
    
    def fix_and_parse_json(self, json_content: str, original_content: str) -> Dict:
        """Attempt to fix common JSON formatting issues and parse"""
        import re
        
        # Try to create a valid JSON structure from common patterns
        try:
            # Extract key information using regex - handle both correct and malformed JSON
            param_match = re.search(r'"parameter":\s*"([^"]*)"', json_content)
            
            # Handle missing "location" key - look for standalone quoted location
            location_match = re.search(r'"location":\s*"([^"]*)"', json_content)
            if not location_match:
                # Look for pattern like: "parameter": "salinity", "Saanich Inlet",
                standalone_location = re.search(r'"parameter":\s*"[^"]*",\s*"([^"]*)",', json_content)
                if standalone_location:
                    location_match = standalone_location
            
            temporal_match = re.search(r'"temporal_reference":\s*"([^"]*)"', json_content)
            type_match = re.search(r'"temporal_type":\s*"([^"]*)"', json_content)
            depth_match = re.search(r'"depth_meters":\s*(null|\d+)', json_content)
            
            # Build a valid JSON structure
            result = {}
            if param_match:
                result["parameter"] = param_match.group(1)
            if location_match:
                result["location"] = location_match.group(1)
            if temporal_match:
                result["temporal_reference"] = temporal_match.group(1)
            if type_match:
                result["temporal_type"] = type_match.group(1)
            else:
                result["temporal_type"] = "single_date"
            if depth_match:
                depth_val = depth_match.group(1)
                result["depth_meters"] = None if depth_val == "null" else int(depth_val)
            else:
                result["depth_meters"] = None
            
            # Check if we have at least parameter and location
            if result.get("parameter") and result.get("location"):
                return result
            
            # If we can't fix it, return empty dict to trigger error handling
            return {}
            
        except Exception as e:
            print(f"Warning: Error in JSON repair: {e}")
            return {}
    
    def parse_temporal_reference(self, temporal_ref: str, temporal_type: str) -> Tuple[datetime, datetime]:
        """Convert natural language dates to datetime objects"""
        if not temporal_ref:
            # Default to last 24 hours
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            return start_time, end_time
        
        now = datetime.now()
        temporal_ref = temporal_ref.lower().strip()
        
        # Handle relative dates
        if "today" in temporal_ref:
            date = now.date()
        elif "yesterday" in temporal_ref:
            date = (now - timedelta(days=1)).date()
        elif "tomorrow" in temporal_ref:
            date = (now + timedelta(days=1)).date()
        else:
            # Handle day names with past/future context
            days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            day_found = False
            for i, day in enumerate(days):
                if day in temporal_ref:
                    current_weekday = now.weekday()
                    days_diff = i - current_weekday
                    
                    # Determine if user wants past or future
                    is_past_reference = any(word in temporal_ref for word in ["last", "previous", "past", "ago"])
                    is_future_reference = any(word in temporal_ref for word in ["next", "coming", "upcoming"])
                    
                    if is_past_reference:
                        # Always go to previous occurrence
                        if days_diff >= 0:
                            days_diff -= 7
                    elif is_future_reference:
                        # Always go to next occurrence
                        if days_diff <= 0:
                            days_diff += 7
                    else:
                        # No explicit past/future - use context clues or default to past for same/past days
                        if days_diff <= 0:
                            days_diff -= 7  # Default to previous week for same day or past days
                    
                    date = (now + timedelta(days=days_diff)).date()
                    day_found = True
                    break
            
            if not day_found:
                # Handle month + day format or month-only format
                months = ["january", "february", "march", "april", "may", "june", 
                         "july", "august", "september", "october", "november", "december"]
                month_found = False
                for i, month in enumerate(months):
                    if month in temporal_ref:
                        import re
                        
                        # Check for specific day in the month
                        day_match = re.search(r'\b(\d{1,2})\b', temporal_ref)
                        if day_match:
                            day = int(day_match.group(1))
                            try:
                                date = datetime(now.year, i+1, day).date()
                                month_found = True
                                break
                            except ValueError:
                                pass  # Invalid date, continue to month-only handling
                        
                        # If no specific day or invalid day, treat as whole month
                        try:
                            # For month queries, set temporal_type to range and return month start/end
                            month_start = datetime(now.year, i+1, 1).date()
                            # Get last day of month
                            if i+1 == 12:  # December
                                month_end = datetime(now.year + 1, 1, 1).date() - timedelta(days=1)
                            else:
                                month_end = datetime(now.year, i+2, 1).date() - timedelta(days=1)
                            
                            # Return month range immediately
                            start_time = datetime.combine(month_start, datetime.min.time())
                            end_time = datetime.combine(month_end, datetime.max.time().replace(microsecond=0))
                            return start_time, end_time
                        except ValueError:
                            pass
                        
                        month_found = True
                        break
                
                if not month_found:
                    # Default to today
                    date = now.date()
        
        # For single dates, check if specific time is mentioned
        if temporal_type == "single_date":
            specific_time = self.extract_specific_time(temporal_ref)
            if specific_time:
                # Use specific time with 1-hour window
                start_time = datetime.combine(date, specific_time)
                end_time = start_time + timedelta(hours=1)
            else:
                # Default to noon (12:00) with 1-hour window for day queries
                from datetime import time
                noon_time = time(12, 0)
                start_time = datetime.combine(date, noon_time)
                end_time = start_time + timedelta(hours=1)
        else:
            # For ranges, would need more sophisticated parsing
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=7)  # Default to week
        
        return start_time, end_time
    
    def extract_specific_time(self, temporal_ref: str) -> Optional[datetime.time]:
        """Extract specific time from temporal reference"""
        from datetime import time
        
        if not temporal_ref:
            return None
        
        temporal_ref = temporal_ref.lower().strip()
        
        # Handle common time words
        if "noon" in temporal_ref or "12 pm" in temporal_ref:
            return time(12, 0)
        elif "midnight" in temporal_ref or "12 am" in temporal_ref:
            return time(0, 0)
        elif "morning" in temporal_ref:
            return time(9, 0)  # 9 AM
        elif "afternoon" in temporal_ref:
            return time(14, 0)  # 2 PM
        elif "evening" in temporal_ref:
            return time(18, 0)  # 6 PM
        elif "night" in temporal_ref:
            return time(21, 0)  # 9 PM
        
        # Handle specific times like "2 pm", "14:00", etc.
        import re
        
        # Pattern for "X pm" or "X am"
        am_pm_pattern = r'(\d{1,2})\s*(am|pm)'
        match = re.search(am_pm_pattern, temporal_ref)
        if match:
            hour = int(match.group(1))
            am_pm = match.group(2)
            if am_pm == 'pm' and hour != 12:
                hour += 12
            elif am_pm == 'am' and hour == 12:
                hour = 0
            return time(hour, 0)
        
        # Pattern for "HH:MM" format
        time_pattern = r'(\d{1,2}):(\d{2})'
        match = re.search(time_pattern, temporal_ref)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return time(hour, minute)
        
        # Pattern for just hour numbers (assume 24-hour format if > 12, otherwise PM)
        hour_pattern = r'\b(\d{1,2})\b'
        matches = re.findall(hour_pattern, temporal_ref)
        if matches:
            hour = int(matches[0])
            if 1 <= hour <= 12 and "o'clock" in temporal_ref:
                # Assume PM for afternoon context
                if hour != 12:
                    hour += 12
                return time(hour, 0)
            elif 0 <= hour <= 23:
                return time(hour, 0)
        
        return None
    
    def is_day_query_with_default_time(self, temporal_ref: str, start_time: datetime, end_time: datetime) -> bool:
        """Check if this was a day query that defaulted to noon"""
        if not temporal_ref:
            return False
        
        temporal_ref = temporal_ref.lower().strip()
        
        # Check if it's a day-level query
        day_indicators = ["today", "yesterday", "tomorrow", "monday", "tuesday", "wednesday", 
                         "thursday", "friday", "saturday", "sunday"]
        is_day_query = any(day in temporal_ref for day in day_indicators)
        
        # Check if we used the noon default (12:00 to 13:00)
        is_noon_default = (start_time.hour == 12 and start_time.minute == 0 and 
                          end_time.hour == 13 and end_time.minute == 0 and 
                          (end_time - start_time).total_seconds() == 3600)
        
        # Check if time was not explicitly specified
        has_explicit_time = any(time_word in temporal_ref for time_word in 
                               ["morning", "afternoon", "evening", "night", "noon", "midnight", 
                                "am", "pm", "hour", "o'clock", ":"])
        
        return is_day_query and is_noon_default and not has_explicit_time
    
    def normalize_parameter(self, param: str) -> Optional[str]:
        """Normalize parameter name"""
        if not param:
            return None
        
        param_lower = param.lower()
        
        # Check aliases
        if param_lower in self.parameter_aliases:
            return self.parameter_aliases[param_lower]
        
        # Check if it's already valid
        valid_params = ["temperature", "salinity", "pressure", "oxygen", 
                       "chlorophyll", "turbidity", "ph", "conductivity"]
        if param_lower in valid_params:
            return param_lower
        
        return None
    
    def normalize_location(self, location: str) -> Optional[str]:
        """Normalize location name"""
        if not location:
            return None
        
        location_lower = location.lower()
        
        # Check known locations
        if location_lower in self.locations:
            return self.locations[location_lower]
        
        # Otherwise return as-is (might be valid)
        return location
    
    def needs_time_clarification(self, temporal_ref: str) -> bool:
        """Check if the temporal reference needs more specific time information"""
        if not temporal_ref:
            return False
        
        temporal_ref = temporal_ref.lower().strip()
        
        # Check for month-level queries that don't need time clarification
        month_indicators = ["month of", "entire month", "whole month", "full month", "throughout", "during"]
        if any(indicator in temporal_ref for indicator in month_indicators):
            return False
        
        # Check for specific month names with broader context
        months = ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"]
        has_month = any(month in temporal_ref for month in months)
        
        # If it's just a month name without a specific day, treat as month-level query
        if has_month:
            # Check if it includes specific day indicators
            day_indicators = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
                            "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th",
                            "21st", "22nd", "23rd", "24th", "25th", "26th", "27th", "28th", "29th", "30th", "31st"]
            has_specific_day = any(day in temporal_ref for day in day_indicators) or \
                             any(f" {i} " in temporal_ref or f" {i}," in temporal_ref for i in range(1, 32))
            
            if not has_specific_day:
                return False  # Month-level query, no clarification needed
        
        # Check for day names without specific time
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        has_day = any(day in temporal_ref for day in days)
        
        # Check for date mentions without time
        has_date_only = any(word in temporal_ref for word in ["today", "yesterday", "tomorrow"])
        
        # Check if time is already specified
        has_time = any(time_word in temporal_ref for time_word in ["morning", "afternoon", "evening", "night", "noon", "midnight", "am", "pm", "hour", "o'clock"])
        
        # Only ask for clarification for specific days without times
        return (has_day or has_date_only) and not has_time
    
    def is_time_response(self, query: str) -> bool:
        """Check if the query is a time specification response"""
        query = query.lower().strip()
        
        # Common time responses
        time_patterns = [
            "morning", "afternoon", "evening", "night", "noon", "midnight",
            "am", "pm", "o'clock", "hour", ":00", ":15", ":30", ":45"
        ]
        
        # Check if it's primarily a time specification
        word_count = len(query.split())
        has_time_words = any(pattern in query for pattern in time_patterns)
        
        # If it's a short response with time words, likely a time specification
        return has_time_words and word_count <= 3
    
    def combine_with_clarification(self, time_response: str) -> Dict:
        """Combine pending partial parameters with time clarification"""
        if not self.pending_clarification:
            return {
                "status": "error",
                "message": "No pending query to complete. Please provide a complete query."
            }
        
        # Get the partial parameters
        partial = self.pending_clarification["partial_parameters"]
        original_temporal_ref = partial["temporal_reference"]
        
        # Combine original temporal reference with time specification
        combined_temporal_ref = f"{original_temporal_ref} {time_response.strip()}"
        
        # Clear pending clarification
        pending_data = self.pending_clarification
        self.pending_clarification = None
        
        # Parse the combined temporal reference
        start_time, end_time = self.parse_temporal_reference(combined_temporal_ref, "single_date")
        
        # Build complete result
        result = {
            "status": "success",
            "parameters": {
                "instrument_type": partial["instrument_type"],
                "location": partial["location"],
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "time_range_hours": (end_time - start_time).total_seconds() / 3600,
                "depth_meters": None
            },
            "metadata": {
                "original_query": f"{pending_data.get('original_query', '')} -> {time_response}",
                "combined_temporal_reference": combined_temporal_ref,
                "model_used": self.model
            }
        }
        
        return result
    
    def extract_parameters(self, query: str) -> Dict:
        """Main method to extract parameters from query"""
        
        # Step 1: Call Groq API
        print("Calling Groq Llama 3 70B...")
        raw_params = self.extract_with_llm(query)
        
        if not raw_params:
            return {
                "status": "error",
                "message": "Failed to extract parameters from query"
            }
        
        # Step 2: Normalize and validate
        parameter = self.normalize_parameter(raw_params.get("parameter"))
        location = self.normalize_location(raw_params.get("location"))
        
        if not parameter:
            return {
                "status": "error",
                "message": "Could not identify measurement type. Please specify: temperature, salinity, pressure, etc."
            }
        
        if not location:
            return {
                "status": "error",
                "message": "Could not identify location. Please specify a location like Cambridge Bay."
            }
        
        # Step 3: Parse dates (no more clarification requests)
        temporal_ref = raw_params.get("temporal_reference", "")
        temporal_type = raw_params.get("temporal_type", "single_date")
        
        start_time, end_time = self.parse_temporal_reference(temporal_ref, temporal_type)
        
        # Step 5: Build result
        result = {
            "status": "success",
            "parameters": {
                "instrument_type": parameter,
                "location": location,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "time_range_hours": (end_time - start_time).total_seconds() / 3600,
                "depth_meters": raw_params.get("depth_meters")
            },
            "metadata": {
                "original_query": query,
                "raw_extraction": raw_params,
                "model_used": self.model
            }
        }
        
        return result
    
    def run_interactive(self):
        """Run in interactive mode"""
        print("Ocean Data Query Parameter Extractor")
        print("Using Groq API with Llama 3 70B")
        print("=" * 60)
        print("\nExample queries:")
        print("  - What is the temperature in Cambridge Bay on January 15?")
        print("  - Show me salinity data for Saanich Inlet yesterday")
        print("  - How cold does it get in Cambridge Bay?")
        print("\nType 'quit' to exit\n")
        
        while True:
            try:
                query = input("\nEnter query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                # Check if this is a time response to a pending clarification
                if self.pending_clarification and self.is_time_response(query):
                    result = self.combine_with_clarification(query)
                else:
                    # Extract parameters normally
                    result = self.extract_parameters(query)
                
                # Display results
                print("\nExtracted Parameters:")
                print("=" * 60)
                
                if result["status"] == "clarification_needed":
                    print(result["message"])
                    continue
                elif result["status"] == "error":
                    print(f"Error: {result['message']}")
                    continue
                
                print(json.dumps(result, indent=2, default=str))
                
                if result["status"] == "success":
                    params = result["parameters"]
                    print(f"\nReady for API call:")
                    print(f"   Instrument: {params['instrument_type']}")
                    print(f"   Location: {params['location']}")
                    print(f"   Start: {params['start_time']}")
                    print(f"   End: {params['end_time']}")
                    print(f"   Duration: {params['time_range_hours']} hours")
                    if params.get('depth_meters'):
                        print(f"   Depth: {params['depth_meters']} meters")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point"""
    try:
        extractor = OceanQueryExtractorGroq()
        
        if len(sys.argv) > 1:
            # Process single query from command line
            query = " ".join(sys.argv[1:])
            result = extractor.extract_parameters(query)
            print(json.dumps(result, indent=2, default=str))
        else:
            # Run in interactive mode
            extractor.run_interactive()
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have set the GROQ_API_KEY environment variable:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        sys.exit(1)


if __name__ == "__main__":
    main()