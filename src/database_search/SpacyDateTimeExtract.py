import spacy
import re
from datetime import datetime, timedelta
from dateutil import parser
import warnings
warnings.filterwarnings("ignore")

class SpacyTimeExtractor:
    def __init__(self):
        """
        Initialize spaCy model
        Run: python -m spacy download en_core_web_sm (if not installed)
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Time patterns for fallback
        self.time_patterns = {
            'time_12h': r'\b(\d{1,2}):?(\d{0,2})\s*(am|pm|AM|PM)\b',
            'time_24h': r'\b(\d{1,2}):(\d{2})\b',
            'time_simple': r'\b(\d{1,2})\s*(o\'?clock|oclock)?\b',
            'range_connectors': r'\b(to|until|till|through|from|between|and|-)\b'
        }
        
        # Date patterns (removed future references)
        self.date_patterns = {
            'relative_days': r'\b(today|yesterday)\b',  # Removed 'tomorrow'
            'weekdays': r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b',
            'date_formats': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'month_day': r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b'
        }
        
        # Future keywords to detect and reject
        self.future_keywords = [
            'next', 'tomorrow', 'future', 'upcoming', 'later', 'will be', 'going to'
        ]
        
        # Time keywords
        self.time_keywords = {
            'morning': ['morning', 'dawn', 'sunrise', 'early'],
            'afternoon': ['afternoon', 'noon', 'midday', 'lunch'],
            'evening': ['evening', 'dusk', 'sunset'],
            'night': ['night', 'midnight', 'late'],
            'now': ['now', 'current', 'currently', 'present', 'right now']
        }

    def _contains_future_reference(self, query):
        """Check if query contains future time references"""
        query_lower = query.lower()
        
        # Check for explicit future keywords
        for keyword in self.future_keywords:
            if keyword in query_lower:
                return True
        
        # Check for specific future patterns
        future_patterns = [
            r'\bnext\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\btomorrow\b',
            r'\bwill\s+be\b',
            r'\bgoing\s+to\b',
            r'\bupcoming\b',
            r'\bin\s+the\s+future\b'
        ]
        
        for pattern in future_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False

    def extract_time_info(self, query):
        """
        Extract time and date information using spaCy and patterns
        """
        result = {
            'has_time_range': False,
            'start_time': None,
            'end_time': None,
            'time_type': 'current',
            'entities': [],
            'message': 'No time information found'
        }
        
        # Check for future references first
        if self._contains_future_reference(query):
            result['message'] = 'Future time references are not supported'
            return result
        
        if not self.nlp:
            return self._fallback_extraction(query)
        
        try:
            # Process with spaCy
            doc = self.nlp(query.lower())
            
            # Extract TIME and DATE entities
            time_entities = []
            date_entities = []
            
            for ent in doc.ents:
                if ent.label_ == 'TIME':
                    time_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
                elif ent.label_ == 'DATE':
                    date_entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            result['entities'] = time_entities + date_entities
            
            # Get base date from date entities or patterns
            base_date = self._extract_date(query, date_entities)
            
            # Check if extracted date is in the future
            if base_date and base_date.date() > datetime.now().date():
                result['message'] = 'Future dates are not supported'
                return result
            
            if time_entities:
                result = self._process_spacy_entities(query, time_entities, result, base_date)
            else:
                # Fallback to pattern matching
                result = self._pattern_extraction(query, result, base_date)
            
        except Exception as e:
            print(f"spaCy processing error: {e}")
            result = self._fallback_extraction(query)
        
        return result
    
    def _extract_date(self, query, date_entities):
        """Extract date information and return base date (only past/present)"""
        query_lower = query.lower()
        
        # Check spaCy date entities first
        if date_entities:
            try:
                date_text = date_entities[0]['text']
                return self._parse_date_text(date_text)
            except:
                pass
        
        # Check for relative date patterns (only past/present)
        if re.search(r'\btoday\b', query_lower):
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif re.search(r'\byesterday\b', query_lower):
            return (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check for "last" + weekday patterns only
        last_weekday_match = re.search(r'\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b', query_lower)
        if last_weekday_match:
            weekday_name = last_weekday_match.group(1)
            weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            weekdays_short = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
            
            if weekday_name in weekdays:
                weekday_index = weekdays.index(weekday_name)
            else:
                weekday_index = weekdays_short.index(weekday_name)
            
            return self._get_last_weekday(weekday_index)
        
        # Check for standalone weekday references (only if it's current week and not future)
        weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        for i, day in enumerate(weekdays):
            if re.search(r'\b' + day + r'\b', query_lower) or re.search(r'\b' + day[:3] + r'\b', query_lower):
                target_date = self._get_current_week_day(i)
                # Only return if it's today or in the past
                if target_date.date() <= datetime.now().date():
                    return target_date
        
        # Check for date patterns (only parse if not in future)
        date_match = re.search(self.date_patterns['date_formats'], query_lower)
        if date_match:
            try:
                parsed_date = parser.parse(date_match.group()).replace(hour=0, minute=0, second=0, microsecond=0)
                if parsed_date.date() <= datetime.now().date():
                    return parsed_date
            except:
                pass
        
        # Check for month-day patterns (only if not in future)
        month_match = re.search(self.date_patterns['month_day'], query_lower)
        if month_match:
            try:
                parsed_date = parser.parse(month_match.group()).replace(hour=0, minute=0, second=0, microsecond=0)
                if parsed_date.date() <= datetime.now().date():
                    return parsed_date
            except:
                pass
        
        # Default to today
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_current_week_day(self, weekday):
        """Get the current week's occurrence of a weekday (0=Monday, 6=Sunday)"""
        today = datetime.now()
        days_diff = weekday - today.weekday()
        return (today + timedelta(days=days_diff)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_last_weekday(self, weekday):
        """Get the last occurrence of a weekday (0=Monday, 6=Sunday)"""
        today = datetime.now()
        days_behind = today.weekday() - weekday
        if days_behind <= 0:  # Target day hasn't happened this week
            days_behind += 7
        return (today - timedelta(days=days_behind)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _parse_date_text(self, date_text):
        """Parse date text to datetime (only past/present)"""
        try:
            date_text = date_text.strip().lower()
            
            if date_text == 'today':
                return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            elif date_text == 'yesterday':
                return (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Handle "last monday", "last friday" etc. only
            last_weekday_match = re.search(r'last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)', date_text)
            if last_weekday_match:
                weekday_name = last_weekday_match.group(1)
                weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
                weekdays_short = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
                
                if weekday_name in weekdays:
                    weekday_index = weekdays.index(weekday_name)
                else:
                    weekday_index = weekdays_short.index(weekday_name)
                
                return self._get_last_weekday(weekday_index)
            
            # Try to parse with dateutil (only if not future)
            parsed_date = parser.parse(date_text)
            parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Check if it's in the future
            if parsed_date.date() > datetime.now().date():
                return None  # Don't return future dates
            
            return parsed_date
            
        except:
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _process_spacy_entities(self, query, entities, result, base_date):
        """Process spaCy detected time entities"""
        try:
            if len(entities) >= 2:
                # Multiple time entities found - likely a range
                result['has_time_range'] = True
                result['time_type'] = 'range'
                
                start_text = entities[0]['text']
                end_text = entities[1]['text']
                
                start_time = self._parse_time_text(start_text)
                end_time = self._parse_time_text(end_text)
                
                # Combine with base date
                result['start_time'] = self._combine_date_time(base_date, start_time)
                result['end_time'] = self._combine_date_time(base_date, end_time)
                result['message'] = f'Found time range: {start_text} to {end_text}'
                
            elif len(entities) == 1:
                # Single time entity
                result['time_type'] = 'specific'
                
                time_text = entities[0]['text']
                parsed_time = self._parse_time_text(time_text)
                result['start_time'] = self._combine_date_time(base_date, parsed_time)
                result['message'] = f'Found specific time: {time_text}'
            
        except Exception as e:
            print(f"Entity processing error: {e}")
            result = self._pattern_extraction(query, result, base_date)
        
        return result
    
    def _combine_date_time(self, base_date, time_obj):
        """Combine base date with time"""
        if isinstance(time_obj, datetime):
            return base_date.replace(
                hour=time_obj.hour,
                minute=time_obj.minute,
                second=time_obj.second,
                microsecond=time_obj.microsecond
            )
        return base_date
    
    def _pattern_extraction(self, query, result, base_date):
        """Pattern-based extraction as fallback"""
        query = query.lower()
        
        # Look for time patterns
        times_found = []
        
        # 12-hour format (improved)
        matches_12h = re.finditer(r'\b(\d{1,2}):?(\d{0,2})\s*(am|pm)\b', query)
        for match in matches_12h:
            times_found.append(match.group())
        
        # 24-hour format
        matches_24h = re.finditer(r'\b(\d{1,2}):(\d{2})\b', query)
        for match in matches_24h:
            hour, minute = match.groups()
            if 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59:
                times_found.append(match.group())
        
        # Simple hour format
        if not times_found:
            matches_simple = re.finditer(r'\b(\d{1,2})\s*(o\'?clock|oclock)?\b', query)
            for match in matches_simple:
                num = int(match.group(1))
                if 1 <= num <= 24:
                    times_found.append(f"{num}:00")
        
        # Check for range indicators
        has_range_indicator = bool(re.search(self.time_patterns['range_connectors'], query))
        
        if len(times_found) >= 2 and has_range_indicator:
            result['has_time_range'] = True
            result['time_type'] = 'range'
            
            start_time = self._parse_time_text(times_found[0])
            end_time = self._parse_time_text(times_found[1])
            
            result['start_time'] = self._combine_date_time(base_date, start_time)
            result['end_time'] = self._combine_date_time(base_date, end_time)
            result['message'] = f'Pattern found range: {times_found[0]} to {times_found[1]}'
            
        elif len(times_found) >= 1:
            result['time_type'] = 'specific'
            parsed_time = self._parse_time_text(times_found[0])
            result['start_time'] = self._combine_date_time(base_date, parsed_time)
            result['message'] = f'Pattern found time: {times_found[0]}'
            
        else:
            # Check for time keywords
            result = self._keyword_extraction(query, result, base_date)
        
        return result
    
    def _keyword_extraction(self, query, result, base_date):
        """Extract time from keywords"""
        for category, keywords in self.time_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    result['time_type'] = 'keyword'
                    keyword_time = self._get_keyword_time(category)
                    result['start_time'] = self._combine_date_time(base_date, keyword_time)
                    result['message'] = f'Found keyword: {keyword}'
                    return result
        
        # Default to current time if no specific time found
        result['start_time'] = datetime.now()
        result['message'] = 'No specific time found, using current time'
        return result
    
    def _parse_time_text(self, time_text):
        """Parse various time text formats"""
        try:
            # Clean the text
            time_text = time_text.strip().lower()
            
            # Handle special cases
            if 'noon' in time_text or 'midday' in time_text:
                return datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
            elif 'midnight' in time_text:
                return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Handle AM/PM format
            if 'am' in time_text or 'pm' in time_text:
                # Extract hour and minute
                time_match = re.search(r'(\d{1,2}):?(\d{0,2})', time_text)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    
                    if 'pm' in time_text and hour != 12:
                        hour += 12
                    elif 'am' in time_text and hour == 12:
                        hour = 0
                    
                    return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # Try to parse with dateutil
            parsed_time = parser.parse(time_text, default=datetime.now())
            return parsed_time
            
        except:
            # Fallback: extract numbers and assume it's hour
            numbers = re.findall(r'\d+', time_text)
            if numbers:
                hour = min(int(numbers[0]), 23)
                return datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)
            
            return datetime.now()
    
    def _get_keyword_time(self, keyword):
        """Convert keyword to datetime"""
        now = datetime.now()
        
        keyword_times = {
            'morning': now.replace(hour=8, minute=0, second=0, microsecond=0),
            'afternoon': now.replace(hour=14, minute=0, second=0, microsecond=0),
            'evening': now.replace(hour=18, minute=0, second=0, microsecond=0),
            'night': now.replace(hour=22, minute=0, second=0, microsecond=0),
            'now': now
        }
        
        return keyword_times.get(keyword, now)
    
    def _fallback_extraction(self, query):
        """Simple fallback when spaCy is not available"""
        result = {
            'has_time_range': False,
            'start_time': datetime.now(),
            'end_time': None,
            'time_type': 'current',
            'entities': [],
            'message': 'Using fallback - spaCy not available'
        }
        
        # Check for future references first
        if self._contains_future_reference(query):
            result['message'] = 'Future time references are not supported'
            result['start_time'] = None
            return result
        
        # Simple number extraction for time ranges
        numbers = re.findall(r'\b\d{1,2}\b', query)
        if len(numbers) >= 2 and any(word in query.lower() for word in ['to', 'until', 'from', 'between']):
            result['has_time_range'] = True
            result['start_time'] = datetime.now().replace(hour=min(int(numbers[0]), 23), minute=0)
            result['end_time'] = datetime.now().replace(hour=min(int(numbers[1]), 23), minute=0)
            result['message'] = f'Fallback found range: {numbers[0]} to {numbers[1]}'
        
        return result

# Simple interface functions
def extract_time_range(query):
    """
    Simple function to get time range from query
    Returns: (has_range, start_time, end_time)
    """
    extractor = SpacyTimeExtractor()
    result = extractor.extract_time_info(query)
    return result['has_time_range'], result['start_time'], result['end_time']

def get_time_info(query):
    """
    Get detailed time information
    Returns: complete result dict
    """
    extractor = SpacyTimeExtractor()
    return extractor.extract_time_info(query)

# Test function
def test_queries():
    """Test with various English queries"""
    
    test_cases = [
        "Show me data from 9 AM to 5 PM",
        "What's the weather like in the morning?",
        "Schedule a meeting at 3:30 PM tomorrow",  # Should be rejected
        "Between 10 and 12 we have lunch today",
        "Current temperature please",
        "Yesterday's report from 2 to 4",
        "Meeting at noon tomorrow",  # Should be rejected
        "Data from last night",
        "From 8 o'clock until midnight",
        "Show me Monday's data at 2 PM",  # Only if Monday is not future
        "What happened on January 15th at 3:30?",
        "Next Friday at 10 AM",  # Should be rejected
        "Last Friday's meeting at 2 PM",
        "Some random text without time info"
    ]
    
    print("=== SpaCy Time Extraction Results (No Future Times) ===\n")
    
    for query in test_cases:
        print(f"Query: '{query}'")
        result = get_time_info(query)
        
        print(f"  Type: {result['time_type']}")
        print(f"  Has Range: {result['has_time_range']}")
        print(f"  Message: {result['message']}")
        
        if result['start_time']:
            print(f"  Start: {result['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if result['end_time']:
            print(f"  End: {result['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        if result['entities']:
            print(f"  Entities: {[ent['text'] for ent in result['entities']]}")
        
        print("-" * 50)

if __name__ == "__main__":
    # Run tests
    test_queries()
    
    # Interactive testing
    print("\n=== Interactive Testing ===")
    while True:
        user_query = input("\nEnter your query (or 'quit'): ")
        
        if user_query.lower() == 'quit':
            break
        
        has_range, start, end = extract_time_range(user_query)
        
        print(f"\nHas time range: {has_range}")
        if start:
            print(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        if end:
            print(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show detailed info
        detailed = get_time_info(user_query)
        print(f"Message: {detailed['message']}")