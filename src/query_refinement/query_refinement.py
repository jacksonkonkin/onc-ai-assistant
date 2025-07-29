"""
Query refinement and assistance module.
Handles ambiguous queries, search suggestions, and feedback collection.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryClarityLevel(Enum):
    """Clarity levels for user queries."""
    CLEAR = "clear"
    SOMEWHAT_AMBIGUOUS = "somewhat_ambiguous"
    HIGHLY_AMBIGUOUS = "highly_ambiguous"


class ResultVolumeLevel(Enum):
    """Result volume levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXCESSIVE = "excessive"


@dataclass
class QueryAnalysis:
    """Analysis results for a user query."""
    clarity_level: QueryClarityLevel
    confidence: float
    ambiguity_issues: List[str]
    suggested_clarifications: List[str]
    missing_context: List[str]
    is_data_download_query: bool = False
    missing_data_parameters: List[str] = None


@dataclass
class ResultAnalysis:
    """Analysis of search results."""
    volume_level: ResultVolumeLevel
    result_count: int
    suggestions: List[str]
    narrowing_filters: List[Dict[str, Any]]


@dataclass
class FeedbackPrompt:
    """Feedback prompt configuration."""
    message: str
    options: List[str]
    follow_up_questions: List[str]


class QueryRefinementManager:
    """
    Manages query refinement, assistance, and feedback collection.
    """
    
    def __init__(self, llm_wrapper=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize query refinement manager.
        
        Args:
            llm_wrapper: LLM wrapper for generating clarifications
            config: Configuration for refinement behavior
        """
        self.llm_wrapper = llm_wrapper
        self.config = config or {}
        
        # Thresholds for different behaviors
        self.ambiguity_threshold = self.config.get('ambiguity_threshold', 0.6)
        self.excessive_results_threshold = self.config.get('excessive_results_threshold', 50)
        self.high_results_threshold = self.config.get('high_results_threshold', 20)
        self.moderate_results_threshold = self.config.get('moderate_results_threshold', 10)
        
        # Enable/disable features
        self.enable_ambiguity_detection = self.config.get('enable_ambiguity_detection', True)
        self.enable_result_suggestions = self.config.get('enable_result_suggestions', True)
        self.enable_feedback_prompts = self.config.get('enable_feedback_prompts', False)  # Disabled for web interface with buttons
        
        logger.info("Query refinement manager initialized")
    
    def analyze_query_clarity(self, query: str, context: Optional[Dict[str, Any]] = None, parameter_extractor=None) -> QueryAnalysis:
        """
        Analyze query clarity and identify ambiguities.
        
        Args:
            query: User query to analyze
            context: Additional context for analysis
            parameter_extractor: Enhanced parameter extractor for validation
            
        Returns:
            QueryAnalysis object with results
        """
        if not self.enable_ambiguity_detection:
            return QueryAnalysis(
                clarity_level=QueryClarityLevel.CLEAR,
                confidence=1.0,
                ambiguity_issues=[],
                suggested_clarifications=[],
                missing_context=[],
                is_data_download_query=False,
                missing_data_parameters=[]
            )
        
        # Check if this is a data download query
        is_data_query = self._is_data_download_query(query)
        
        # Rule-based ambiguity detection
        ambiguity_issues = self._detect_ambiguity_issues(query)
        missing_context = self._detect_missing_context(query)
        
        # Enhanced validation for data download queries using parameter extractor
        missing_data_parameters = []
        if is_data_query:
            missing_data_parameters = self._validate_data_query_parameters(query, context, parameter_extractor)
            # For data queries, missing parameters should increase ambiguity
            if missing_data_parameters:
                missing_context.extend(missing_data_parameters)
        
        # Calculate clarity level
        clarity_level, confidence = self._calculate_clarity_level(query, ambiguity_issues, missing_context, is_data_query)
        
        # Generate clarification suggestions
        suggested_clarifications = self._generate_clarification_suggestions(
            query, ambiguity_issues, missing_context, context, is_data_query, missing_data_parameters
        )
        
        return QueryAnalysis(
            clarity_level=clarity_level,
            confidence=confidence,
            ambiguity_issues=ambiguity_issues,
            suggested_clarifications=suggested_clarifications,
            missing_context=missing_context,
            is_data_download_query=is_data_query,
            missing_data_parameters=missing_data_parameters
        )
    
    def _detect_ambiguity_issues(self, query: str) -> List[str]:
        """Detect specific ambiguity issues in the query."""
        issues = []
        query_lower = query.lower()
        
        # Vague temporal references
        vague_time_patterns = [
            r'\b(recently|lately|soon|sometimes|often|usually)\b',
            r'\b(last time|this time|next time)\b',
            r'\b(current|now|today)\b(?!\s+(?:temperature|data|reading))'
        ]
        
        for pattern in vague_time_patterns:
            if re.search(pattern, query_lower):
                issues.append("Vague temporal reference")
                break
        
        # Ambiguous location references
        if re.search(r'\b(there|here|that place|this location)\b', query_lower):
            issues.append("Ambiguous location reference")
        
        # Vague parameter references
        vague_param_patterns = [
            r'\b(the (?:sensor|device|instrument|data))\b(?!\s+(?:at|from|in))',
            r'\b(some|any|those|these)\s+(?:sensors|devices|instruments|data)\b'
        ]
        
        for pattern in vague_param_patterns:
            if re.search(pattern, query_lower):
                issues.append("Vague parameter reference")
                break
        
        # Ambiguous pronouns
        if re.search(r'\b(it|this|that|they|them)\b', query_lower):
            issues.append("Ambiguous pronoun usage")
        
        # Incomplete comparisons
        if re.search(r'\b(higher|lower|better|worse|more|less)\b(?!\s+than)', query_lower):
            issues.append("Incomplete comparison")
        
        return issues
    
    def _is_device_discovery_query(self, query: str) -> bool:
        """Check if query is about device discovery."""
        if not query:
            return False
        
        query_lower = query.lower()
        
        # Device discovery patterns (including data products discovery)
        device_discovery_patterns = [
            'what devices', 'what sensors', 'what instruments',
            'show me devices', 'show me sensors', 'show me instruments',
            'find devices', 'find sensors', 'find instruments',
            'list devices', 'list sensors', 'list instruments',
            'devices are available', 'sensors are available', 'instruments are available',
            'devices at cambridge bay', 'sensors at cambridge bay',
            'show me data products'
        ]
        
        # Check for device discovery patterns
        if any(pattern in query_lower for pattern in device_discovery_patterns):
            return True
        
        # Check for data products discovery patterns (more flexible)
        has_data_products = 'data products' in query_lower
        has_what_available = 'what' in query_lower and 'available' in query_lower
        has_show_available = 'show' in query_lower and 'available' in query_lower
        
        if has_data_products and (has_what_available or has_show_available):
            return True
        
        # Device terms
        device_terms = [
            'device', 'sensor', 'instrument', 'equipment',
            'ctd', 'hydrophone', 'oxygen sensor', 'ph sensor',
            'weather station', 'ice profiler', 'camera', 'fluorometer'
        ]
        
        # Check for device terms with availability/discovery context
        has_device_term = any(term in query_lower for term in device_terms)
        has_availability_context = any(term in query_lower for term in [
            'available', 'deployed', 'are there', 'which', 'what'
        ])
        has_location_context = any(term in query_lower for term in [
            'cambridge bay', 'cambridge', 'at', 'in', 'location'
        ])
        
        # If query has device terms with availability and location context, likely device discovery
        if has_device_term and has_availability_context and has_location_context:
            return True
        
        return False

    def _is_data_download_query(self, query: str) -> bool:
        """Determine if this is a database/data query that may need parameter validation."""
        query_lower = query.lower()
        
        # Skip if this is a device discovery query
        if self._is_device_discovery_query(query):
            return False
        
        # Data request patterns - broader than before to catch database queries
        data_request_patterns = [
            r'\b(get|show|give|find|retrieve|download|fetch)\s+(me\s+)?(the\s+)?(data|temperature|salinity|pressure)\b',
            r'\b(what\s+(was|is)\s+the)\s+(temperature|salinity|pressure|ph|oxygen|conductivity)\b',
            r'\b(temperature|salinity|pressure|ph|oxygen|conductivity|chlorophyll|turbidity|density|fluorescence)\s+(data|reading|measurement|value|at|from|in)\b',
            r'\b(data|measurement|reading|value)\s+(from|at|in|for)\b',
            r'\b(latest|current|recent)\s+(temperature|salinity|pressure|ph|oxygen|conductivity)\b',
            r'\b(sensor|instrument)\s+(data|reading|measurement)\b',
            r'\b(how\s+(hot|cold|warm))\b',  # Temperature requests
            r'\b(what.*temperature|tell.*about.*temperature)\b'
        ]
        
        for pattern in data_request_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for parameter names that typically indicate data requests
        parameter_names = [
            'temperature', 'salinity', 'pressure', 'ph', 'oxygen', 'conductivity',
            'chlorophyll', 'turbidity', 'density', 'fluorescence', 'depth'
        ]
        
        # Broader matching - parameter + any context suggests data query
        has_parameter = any(param in query_lower for param in parameter_names)
        has_data_context = bool(re.search(r'\b(cambridge bay|cbyip|station|data|measurement|sensor|yesterday|today|last|this|now|current|at|from|in)\b', query_lower))
        
        # If query contains parameter name + any data context, likely a data query
        if has_parameter and has_data_context:
            return True
        
        # Check for implicit data queries (questions about ocean conditions)
        implicit_data_patterns = [
            r'\bhow\s+(hot|cold|warm|salty)\b',
            r'\bwhat.*temperature.*cambridge bay\b',
            r'\bwhat.*salinity\b',
            r'\bwhat.*pressure\b',
            r'\bcambridge bay.*temperature\b',
            r'\bcambridge bay.*salinity\b',
            r'\bcambridge bay.*pressure\b'
        ]
        
        for pattern in implicit_data_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _validate_data_query_parameters(self, query: str, context: Optional[Dict[str, Any]] = None, parameter_extractor=None) -> List[str]:
        """Validate that a data query has all required parameters using the parameter extractor."""
        missing = []
        
        # Get conversation context to check for previously provided information
        conversation_context = ""
        if context and 'conversation_context' in context:
            conversation_context = context['conversation_context']
        
        # Combined context: current query + conversation history
        combined_context = f"{conversation_context} {query}".strip()
        
        # Try parameter extraction to see what's missing
        if parameter_extractor:
            try:
                extraction_result = parameter_extractor.extract_parameters(combined_context)
                
                if extraction_result.get("status") == "success":
                    params = extraction_result.get("parameters", {})
                    
                    # Check for missing or invalid location
                    location_code = params.get("location_code")
                    if not location_code or location_code not in parameter_extractor.location_devices:
                        missing.append("Specific location (e.g., 'Cambridge Bay', 'CBYIP station', or coordinates)")
                    
                    # Check for missing or invalid property/parameter type
                    property_code = params.get("property_code")
                    device_category = params.get("device_category")
                    if not property_code:
                        missing.append("Specific parameter type (e.g., 'temperature', 'salinity', 'pressure', 'oxygen')")
                    elif device_category and property_code not in parameter_extractor.device_properties.get(device_category, []):
                        missing.append("Valid parameter type for the selected device/location")
                    
                    # Check for missing temporal information
                    start_time = params.get("start_time")
                    end_time = params.get("end_time")
                    temporal_ref = params.get("temporal_reference")
                    
                    if not start_time or not end_time or not temporal_ref:
                        missing.append("Specific date and time (e.g., 'June 25, 2024 at 2:00 PM', 'yesterday at noon', 'last week')")
                    else:
                        # Check if time range is too vague (e.g., whole day without specific hours)
                        try:
                            from datetime import datetime
                            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                            time_diff = end_dt - start_dt
                            
                            # If the time range is more than 24 hours and no specific time was mentioned
                            if time_diff.total_seconds() > 86400:  # More than 1 day
                                has_specific_time = bool(re.search(r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)|morning|afternoon|evening|night|noon|midnight)\b', combined_context.lower()))
                                if not has_specific_time:
                                    missing.append("Specific time or time range (what time of day? e.g., '2:00 PM - 4:00 PM', 'morning', 'afternoon')")
                        except (ValueError, TypeError):
                            # If we can't parse the times, ask for clarification
                            missing.append("Valid date and time format")
                    
                    # Check for device category validation
                    if not device_category or (location_code and device_category not in parameter_extractor.location_devices.get(location_code, [])):
                        # Only ask for device clarification if it's truly ambiguous
                        if property_code and len([d for d in parameter_extractor.location_devices.get(location_code, []) 
                                               if property_code in parameter_extractor.device_properties.get(d, [])]) > 1:
                            missing.append("Specific sensor/device type (multiple devices can measure this parameter)")
                    
                else:
                    # Parameter extraction failed, use fallback validation
                    return self._fallback_parameter_validation(combined_context)
                    
            except Exception as e:
                logger.warning(f"Parameter extraction failed during validation: {e}")
                return self._fallback_parameter_validation(combined_context)
        else:
            # No parameter extractor available, use fallback
            return self._fallback_parameter_validation(combined_context)
        
        return missing
    
    def _fallback_parameter_validation(self, combined_context: str) -> List[str]:
        """Fallback parameter validation when extractor is not available."""
        missing = []
        context_lower = combined_context.lower()
        
        # Check for location
        has_location = bool(re.search(r'\b(cambridge bay|cbyip|station|location|coordinates|latitude|longitude)\b', context_lower))
        if not has_location:
            missing.append("Specific location (e.g., 'Cambridge Bay', 'CBYIP station')")
        
        # Check for parameter type (including implicit references)
        has_parameter = bool(re.search(r'\b(temperature|salinity|pressure|ph|oxygen|conductivity|chlorophyll|turbidity|density|fluorescence|hot|cold|warm|salty)\b', context_lower))
        if not has_parameter:
            missing.append("Specific parameter type (e.g., 'temperature', 'salinity', 'pressure')")
        
        # Check for temporal information
        has_time = bool(re.search(r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|january|february|march|april|may|june|july|august|september|october|november|december|yesterday|today|tomorrow|last|this|now|current|\d{1,2}:\d{2})\b', context_lower))
        if not has_time:
            missing.append("Specific date and time (e.g., 'June 25, 2024', 'yesterday', 'last week')")
        
        return missing
    
    def _check_parameter_specificity(self, query_lower: str) -> List[str]:
        """Check for parameter specificity issues."""
        issues = []
        
        # Temperature ambiguity (air vs water)
        if 'temperature' in query_lower:
            # Check if air vs water temperature is specified
            has_water_context = bool(re.search(r'\b(water|sea|ocean|seawater)\s+temperature\b', query_lower))
            has_air_context = bool(re.search(r'\b(air|atmospheric)\s+temperature\b', query_lower))
            
            if not (has_water_context or has_air_context):
                issues.append("Temperature type (air temperature or water temperature?)")
        
        # Pressure ambiguity (atmospheric vs water)
        if 'pressure' in query_lower:
            has_water_pressure = bool(re.search(r'\b(water|hydrostatic|depth)\s+pressure\b', query_lower))
            has_air_pressure = bool(re.search(r'\b(air|atmospheric|barometric)\s+pressure\b', query_lower))
            
            if not (has_water_pressure or has_air_pressure):
                issues.append("Pressure type (atmospheric pressure or water pressure?)")
        
        # Generic "data" without specific parameter
        if re.search(r'\b(data|measurement|reading|value)\b', query_lower):
            specific_params = ['temperature', 'salinity', 'pressure', 'ph', 'oxygen', 'conductivity', 'chlorophyll']
            has_specific_param = any(param in query_lower for param in specific_params)
            
            if not has_specific_param:
                issues.append("Specific parameter type (temperature, salinity, pressure, etc.)")
        
        return issues
    
    def _check_sensor_specificity(self, query_lower: str) -> Optional[str]:
        """Check if sensor/instrument type needs clarification."""
        
        # Check for generic sensor references
        if re.search(r'\b(sensor|instrument|device)\s+(data|reading|measurement)\b', query_lower):
            specific_sensors = ['ctd', 'thermometer', 'conductivity', 'oxygen', 'fluorometer']
            has_specific_sensor = any(sensor in query_lower for sensor in specific_sensors)
            
            if not has_specific_sensor:
                return "Specific sensor type (CTD, thermometer, oxygen sensor, etc.)"
        
        return None
    
    def _detect_missing_context(self, query: str) -> List[str]:
        """Detect missing context that would help clarify the query."""
        missing = []
        query_lower = query.lower()
        
        # Check for missing time context
        if not re.search(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}:\d{2}|am|pm|yesterday|today|tomorrow|last week|last month|last year)\b', query_lower):
            if re.search(r'\b(data|measurement|reading|temperature|salinity|pressure)\b', query_lower):
                missing.append("Specific time or date")
        
        # Check for missing location context
        if not re.search(r'\b(cambridge bay|cbyip|station|site|location)\b', query_lower):
            if re.search(r'\b(sensor|device|instrument|data|measurement)\b', query_lower):
                missing.append("Specific location or station")
        
        # Check for missing parameter context
        if re.search(r'\b(data|measurement|reading|value)\b', query_lower):
            if not re.search(r'\b(temperature|salinity|pressure|ph|oxygen|conductivity|chlorophyll|turbidity|density|fluorescence)\b', query_lower):
                missing.append("Specific parameter type")
        
        return missing
    
    def _calculate_clarity_level(self, query: str, issues: List[str], missing: List[str], is_data_query: bool = False) -> Tuple[QueryClarityLevel, float]:
        """Calculate overall clarity level and confidence."""
        total_issues = len(issues) + len(missing)
        query_length = len(query.split())
        
        # For data queries, be more strict about missing parameters
        if is_data_query:
            # Data queries with any missing critical parameters should be considered ambiguous
            if total_issues > 0:
                if total_issues >= 3:
                    return QueryClarityLevel.HIGHLY_AMBIGUOUS, 0.9
                elif total_issues >= 2:
                    return QueryClarityLevel.HIGHLY_AMBIGUOUS, 0.8
                else:
                    return QueryClarityLevel.SOMEWHAT_AMBIGUOUS, 0.7
            else:
                return QueryClarityLevel.CLEAR, 0.9
        
        # Original logic for non-data queries
        issue_density = total_issues / max(query_length, 1)
        
        if issue_density > 0.4 or total_issues > 3:
            return QueryClarityLevel.HIGHLY_AMBIGUOUS, 0.8
        elif issue_density > 0.2 or total_issues > 1:
            return QueryClarityLevel.SOMEWHAT_AMBIGUOUS, 0.7
        else:
            return QueryClarityLevel.CLEAR, 0.9
    
    def _generate_clarification_suggestions(self, query: str, issues: List[str], missing: List[str], context: Optional[Dict[str, Any]] = None, is_data_query: bool = False, missing_data_parameters: List[str] = None) -> List[str]:
        """Generate specific clarification suggestions."""
        suggestions = []
        
        # Use LLM if available for intelligent suggestions
        if self.llm_wrapper:
            llm_suggestions = self._generate_llm_clarifications(query, issues, missing, context, is_data_query, missing_data_parameters)
            suggestions.extend(llm_suggestions)
        
        # Fallback to rule-based suggestions
        if not suggestions:
            suggestions = self._generate_rule_based_clarifications(query, issues, missing, is_data_query, missing_data_parameters)
        
        return suggestions[:4]  # Allow up to 4 suggestions for data queries
    
    def _generate_llm_clarifications(self, query: str, issues: List[str], missing: List[str], context: Optional[Dict[str, Any]] = None, is_data_query: bool = False, missing_data_parameters: List[str] = None) -> List[str]:
        """Generate clarification suggestions using LLM."""
        try:
            # Enhanced prompt for data queries
            if is_data_query and missing_data_parameters:
                conversation_context = context.get('conversation_context', '') if context else ''
                
                prompt = f"""You are an expert at helping users refine their oceanographic data queries for Ocean Networks Canada.

User's Current Query: "{query}"

{f'''Previous Conversation Context:
{conversation_context}

''' if conversation_context else ''}This query needs data from Ocean Networks Canada sensors, but some required parameters are missing.

Missing Required Parameters:
{chr(10).join(f"- {param}" for param in missing_data_parameters) if missing_data_parameters else "- None"}

IMPORTANT: Check the conversation context carefully. Only ask for information that is truly missing - don't re-ask for details already provided in previous messages.

Generate 1-3 specific clarifying questions to collect the missing information. Focus on:

1. **Location**: If missing, ask for specific ONC location (e.g., "Cambridge Bay", "CBYIP station")
2. **Time/Date**: If missing, ask for specific date and time period needed
3. **Parameter**: If missing, ask for specific oceanographic parameter (temperature, salinity, etc.)
4. **Time Range**: If date is provided but time range is too broad, ask for specific hours

Make each question conversational and helpful. Format as simple questions, one per line, without bullets or numbers.

IMPORTANT FORMATTING:
- Write ONLY the questions, one per line
- DO NOT add introductory text like "Here are three clarifying questions"
- DO NOT number the questions
- DO NOT add explanatory text

Example good questions:
Which specific location do you need data from? (e.g., Cambridge Bay, CBYIP station)
What date and time period are you interested in? (e.g., June 15, 2024 from 2:00-4:00 PM)
Which oceanographic parameter do you need? (e.g., water temperature, salinity, pressure)

Keep questions natural and specific to oceanographic data collection."""
            else:
                prompt = f"""You are an expert at helping users refine their queries about oceanographic data.

User Query: "{query}"

Identified Issues:
{chr(10).join(f"- {issue}" for issue in issues) if issues else "- None"}

Missing Context:
{chr(10).join(f"- {item}" for item in missing) if missing else "- None"}

Generate 2-3 specific clarifying questions that would help make this query more precise and answerable. Focus on:
1. Specific locations (e.g., "Cambridge Bay", "CBYIP station")
2. Specific time periods (e.g., "October 2022", "last week")
3. Specific parameters (e.g., "temperature", "salinity", "pressure")

Format as a simple list, one question per line, without bullet points or numbers.
Keep questions concise and natural."""

            response = self.llm_wrapper.invoke(prompt)
            
            # Parse response into individual questions
            questions = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('-')]
            return questions[:4] if is_data_query else questions[:3]
            
        except Exception as e:
            logger.warning(f"LLM clarification generation failed: {e}")
            return []
    
    def _generate_rule_based_clarifications(self, query: str, issues: List[str], missing: List[str], is_data_query: bool = False, missing_data_parameters: List[str] = None) -> List[str]:
        """Generate clarification suggestions using rules."""
        suggestions = []
        
        # Enhanced clarifications for data queries
        if is_data_query and missing_data_parameters:
            # Handle specific data parameter issues
            for param in missing_data_parameters:
                if "Specific location" in param:
                    suggestions.append("Which exact location or station do you need data from? (e.g., 'Cambridge Bay', 'CBYIP station', specific coordinates)")
                elif "Specific date" in param and "exact date" in param:
                    suggestions.append("What specific date are you referring to? Please provide the exact date (e.g., 'June 25, 2024')")
                elif "Specific time period" in param and "recent" in param:
                    suggestions.append("How recent do you mean? Please specify the time period (e.g., 'last week', 'last month', 'past 6 months')")
                elif "Specific current time" in param:
                    suggestions.append("Do you want the most recent available data, or data from a specific current time period? Please clarify the timeframe.")
                elif "Year for the date" in param:
                    suggestions.append("Which year are you asking about? Please specify the year (e.g., '2024', '2023', etc.)")
                elif "Specific time or time range" in param and "what time of day" in param:
                    suggestions.append("What specific time or time range do you need? Please provide hours (e.g., '2:00 PM - 4:00 PM', '9:00 AM', 'morning', 'afternoon')")
                elif "Temperature type" in param:
                    suggestions.append("Do you need air temperature or water temperature data?")
                elif "Pressure type" in param:
                    suggestions.append("Do you need atmospheric pressure or water pressure data?")
                elif "Specific parameter type" in param:
                    suggestions.append("What specific oceanographic parameter do you need? (e.g., 'temperature', 'salinity', 'pressure', 'oxygen', 'pH')")
                elif "Specific sensor type" in param:
                    suggestions.append("Which specific sensor or instrument type? (e.g., 'CTD sensor', 'thermometer', 'oxygen sensor')")
                elif "date and time" in param:
                    suggestions.append("Please provide the specific date and time period you're interested in (e.g., 'June 25, 2024 at 2:00 PM' or 'June 20-25, 2024')")
        
        # General clarifications (fallback and non-data queries)
        if "Specific time or date" in missing:
            suggestions.append("Could you specify when you're interested in? (e.g., 'October 2022', 'last week', 'yesterday')")
        
        if "Specific time or time range" in missing and "what time of day" not in missing:
            suggestions.append("What specific time or time range do you need? Please provide hours (e.g., '2:00 PM - 4:00 PM', '9:00 AM', 'morning', 'afternoon')")
        
        if "Specific location or station" in missing:
            suggestions.append("Which location or station are you asking about? (e.g., 'Cambridge Bay', 'CBYIP station')")
        
        if "Specific parameter type" in missing:
            suggestions.append("What type of data are you looking for? (e.g., 'temperature', 'salinity', 'pressure')")
        
        if "Vague temporal reference" in issues:
            suggestions.append("Could you be more specific about the time period?")
        
        if "Ambiguous location reference" in issues:
            suggestions.append("Could you specify the exact location or station name?")
        
        # Remove duplicates while preserving order and avoid redundant questions
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            # Normalize suggestion for duplicate detection
            normalized = suggestion.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:3]  # Limit to 3 questions maximum
    
    def analyze_results(self, results: List[Any], query: str, search_type: str = "unknown") -> ResultAnalysis:
        """
        Analyze search results and provide suggestions for refinement.
        
        Args:
            results: Search results to analyze
            query: Original query
            search_type: Type of search performed
            
        Returns:
            ResultAnalysis with suggestions
        """
        if not self.enable_result_suggestions:
            return ResultAnalysis(
                volume_level=ResultVolumeLevel.MODERATE,
                result_count=len(results),
                suggestions=[],
                narrowing_filters=[]
            )
        
        result_count = len(results)
        
        # Determine volume level
        if result_count == 0:
            volume_level = ResultVolumeLevel.NONE
        elif result_count <= self.moderate_results_threshold:
            volume_level = ResultVolumeLevel.LOW
        elif result_count <= self.high_results_threshold:
            volume_level = ResultVolumeLevel.MODERATE
        elif result_count <= self.excessive_results_threshold:
            volume_level = ResultVolumeLevel.HIGH
        else:
            volume_level = ResultVolumeLevel.EXCESSIVE
        
        # Generate suggestions based on volume level
        suggestions = self._generate_result_suggestions(volume_level, result_count, query, search_type)
        narrowing_filters = self._generate_narrowing_filters(volume_level, query, search_type)
        
        return ResultAnalysis(
            volume_level=volume_level,
            result_count=result_count,
            suggestions=suggestions,
            narrowing_filters=narrowing_filters
        )
    
    def _generate_result_suggestions(self, volume_level: ResultVolumeLevel, count: int, query: str, search_type: str) -> List[str]:
        """Generate suggestions based on result volume."""
        suggestions = []
        
        if volume_level == ResultVolumeLevel.NONE:
            suggestions.extend([
                "Try broadening your search terms or checking for typos",
                "Consider using more general terms or different parameter names",
                "Check if the location or time period has available data"
            ])
        elif volume_level == ResultVolumeLevel.EXCESSIVE:
            suggestions.extend([
                "Try adding more specific time constraints to narrow your search",
                "Specify a particular location or station to focus your results",
                "Add specific parameter types to filter the data"
            ])
        elif volume_level == ResultVolumeLevel.HIGH:
            suggestions.extend([
                "Consider adding time or location filters to refine your search",
                "You might want to focus on specific parameters or instruments"
            ])
        
        return suggestions
    
    def _generate_narrowing_filters(self, volume_level: ResultVolumeLevel, query: str, search_type: str) -> List[Dict[str, Any]]:
        """Generate specific narrowing filter suggestions."""
        filters = []
        
        if volume_level in [ResultVolumeLevel.HIGH, ResultVolumeLevel.EXCESSIVE]:
            query_lower = query.lower()
            
            # Time-based filters
            if not re.search(r'\b(\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|last|this|yesterday|today)\b', query_lower):
                filters.append({
                    "type": "time",
                    "suggestion": "Add a specific time period",
                    "examples": ["last week", "October 2022", "2023-01-15"]
                })
            
            # Location-based filters
            if not re.search(r'\b(cambridge bay|cbyip|station|site)\b', query_lower):
                filters.append({
                    "type": "location",
                    "suggestion": "Specify a location or station",
                    "examples": ["Cambridge Bay", "CBYIP station"]
                })
            
            # Parameter-based filters
            if not re.search(r'\b(temperature|salinity|pressure|ph|oxygen)\b', query_lower):
                filters.append({
                    "type": "parameter",
                    "suggestion": "Focus on specific parameters",
                    "examples": ["temperature", "salinity", "pressure"]
                })
        
        return filters
    
    def generate_feedback_prompt(self, query: str, response: str, routing_info: Optional[Dict[str, Any]] = None) -> FeedbackPrompt:
        """
        Generate simple feedback prompt for user interaction.
        
        Args:
            query: Original user query
            response: Generated response
            routing_info: Information about how query was routed
            
        Returns:
            FeedbackPrompt object with simple thumbs up/down
        """
        if not self.enable_feedback_prompts:
            return FeedbackPrompt(message="", options=[], follow_up_questions=[])
        
        # Simple thumbs up/down message
        message = "👍 👎 Was this helpful?"
        
        # No verbose options or follow-up questions
        options = []
        follow_up_questions = []
        
        return FeedbackPrompt(
            message=message,
            options=options,
            follow_up_questions=follow_up_questions
        )
    
    def _generate_follow_up_questions(self, query: str, response: str, routing_info: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate contextual follow-up questions."""
        questions = []
        
        # Check if response contains data
        if re.search(r'\b(temperature|salinity|pressure|°c|ppt|dbar)\b', response.lower()):
            questions.extend([
                "Would you like to see this data in a different format?",
                "Are you interested in trends or patterns in this data?",
                "Would you like to compare this with data from other locations or times?"
            ])
        
        # Check if response is about instruments/sensors
        if re.search(r'\b(sensor|instrument|device|deployment)\b', response.lower()):
            questions.extend([
                "Would you like to know more about the technical specifications?",
                "Are you interested in the current status of these instruments?",
                "Would you like to see data from these sensors?"
            ])
        
        # Check if response indicates no data found
        if re.search(r'\b(no data|not available|not found)\b', response.lower()):
            questions.extend([
                "Would you like suggestions for similar data that might be available?",
                "Should I help you refine your search criteria?",
                "Are you interested in data from nearby locations or different time periods?"
            ])
        
        return questions[:2]  # Limit to 2 follow-up questions
    
    def should_request_clarification(self, analysis: QueryAnalysis) -> bool:
        """
        Determine if we should request clarification from the user.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            bool: True if clarification should be requested
        """
        return (
            analysis.clarity_level == QueryClarityLevel.HIGHLY_AMBIGUOUS and
            analysis.confidence >= self.ambiguity_threshold and
            len(analysis.suggested_clarifications) > 0
        )
    
    def should_provide_suggestions(self, analysis: ResultAnalysis) -> bool:
        """
        Determine if we should provide search refinement suggestions.
        
        Args:
            analysis: Result analysis
            
        Returns:
            bool: True if suggestions should be provided
        """
        return (
            analysis.volume_level in [ResultVolumeLevel.HIGH, ResultVolumeLevel.EXCESSIVE, ResultVolumeLevel.NONE] and
            len(analysis.suggestions) > 0
        )
    
    def format_clarification_request(self, analysis: QueryAnalysis) -> str:
        """
        Format a clarification request message.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            Formatted clarification message
        """
        message = "I'd like to help you find the most relevant information. Could you clarify:\n\n"
        
        for i, suggestion in enumerate(analysis.suggested_clarifications, 1):
            message += f"{i}. {suggestion}\n"
        
        message += "\nThis will help me provide you with more accurate and specific results."
        
        return message
    
    def format_result_suggestions(self, analysis: ResultAnalysis) -> str:
        """
        Format result refinement suggestions.
        
        Args:
            analysis: Result analysis
            
        Returns:
            Formatted suggestions message
        """
        if analysis.volume_level == ResultVolumeLevel.NONE:
            return ""
        elif analysis.volume_level in [ResultVolumeLevel.HIGH, ResultVolumeLevel.EXCESSIVE]:
            message = f"I found {analysis.result_count} results. To help narrow this down:\n\n"
        else:
            return ""
        
        for i, suggestion in enumerate(analysis.suggestions, 1):
            message += f"{i}. {suggestion}\n"
        
        return message
    
    def format_feedback_prompt(self, feedback_prompt: FeedbackPrompt) -> str:
        """
        Format simple feedback prompt for display.
        
        Args:
            feedback_prompt: Feedback prompt object
            
        Returns:
            Simple formatted feedback message
        """
        if not feedback_prompt.message:
            return ""
        
        # Simple one-line feedback prompt
        return f"\n\n{feedback_prompt.message}"