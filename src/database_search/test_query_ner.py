#!/usr/bin/env python3
"""
Test script for Ocean Query NER system
Runs 20 diverse test queries to identify failure cases and edge conditions
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path so we can import queryNER
sys.path.append(str(Path(__file__).parent))

from queryNER import OceanQueryExtractorGroq

class QueryNERTester:
    """Test suite for Ocean Query NER system"""
    
    def __init__(self):
        self.extractor = OceanQueryExtractorGroq()
        self.test_queries = [
            # Basic temporal queries
            "What is the temperature in Cambridge Bay today?",
            "Show me salinity data for Saanich Inlet yesterday",
            "How cold was it in Cambridge Bay last Monday?",
            
            # Month-level queries
            "What are the salinity levels in Cambridge Bay for the month of January?",
            "Show me temperature data for Saanich Inlet in February",
            "Get oxygen levels for the entire month of March in Cambridge Bay",
            
            # Specific time queries
            "What was the temperature at Cambridge Bay on January 15th at 2 PM?",
            "Show me salinity at noon on Tuesday in Saanich Inlet",
            "Get pressure data for Cambridge Bay tomorrow morning",
            
            # Week/range queries
            "Temperature data for Cambridge Bay from January 1 to January 7",
            "Show me salinity trends for the past week in Saanich Inlet",
            "Get oxygen levels from last Monday to this Friday",
            
            # Parameter variations
            "How salty is the water in Cambridge Bay right now?",
            "What's the dissolved oxygen content in Saanich Inlet today?",
            "Show me pH levels for Cambridge Bay yesterday",
            "Get turbidity readings for Saanich Inlet this morning",
            
            # Location variations
            "Temperature in Strait of Georgia today",
            "Salinity levels in Iqaluktuuttiaq last week",
            
            # Edge cases and potential failures
            "What's the weather like in Cambridge Bay?",  # Wrong parameter type
            "Temperature data for New York yesterday",     # Wrong location
            "Show me data for Cambridge Bay",              # Missing parameter
            "Get temperature",                             # Missing location
            "How deep is Cambridge Bay?",                  # Not a measurement query
        ]
        
        self.results = []
        
    def run_test(self, query, test_number):
        """Run a single test query and capture results"""
        print(f"\n{'='*80}")
        print(f"TEST {test_number}: {query}")
        print(f"{'='*80}")
        
        try:
            result = self.extractor.extract_parameters(query)
            
            test_result = {
                "test_number": test_number,
                "query": query,
                "status": result.get("status", "unknown"),
                "success": result.get("status") == "success",
                "needs_clarification": result.get("status") == "clarification_needed",
                "error": result.get("status") == "error",
                "error_message": result.get("message") if result.get("status") == "error" else None,
                "clarification_message": result.get("message") if result.get("status") == "clarification_needed" else None,
                "parameters": result.get("parameters"),
                "raw_result": result
            }
            
            # Display result
            if result["status"] == "success":
                params = result["parameters"]
                print(f"SUCCESS")
                print(f"   Parameter: {params['instrument_type']}")
                print(f"   Location: {params['location']}")
                print(f"   Time Range: {params['start_time']} to {params['end_time']}")
                print(f"   Duration: {params['time_range_hours']:.1f} hours")
                if params.get('depth_meters'):
                    print(f"   Depth: {params['depth_meters']} meters")
                    
            elif result["status"] == "clarification_needed":
                print(f"CLARIFICATION NEEDED")
                print(f"   Message: {result['message']}")
                
            elif result["status"] == "error":
                print(f"ERROR")
                print(f"   Message: {result['message']}")
                
            else:
                print(f"UNKNOWN STATUS: {result['status']}")
                
        except Exception as e:
            print(f"EXCEPTION: {str(e)}")
            test_result = {
                "test_number": test_number,
                "query": query,
                "status": "exception",
                "success": False,
                "needs_clarification": False,
                "error": True,
                "error_message": str(e),
                "exception": True,
                "raw_result": None
            }
        
        self.results.append(test_result)
        return test_result
    
    def run_all_tests(self):
        """Run all test queries"""
        print("STARTING OCEAN QUERY NER TEST SUITE")
        print(f"Running {len(self.test_queries)} test queries...")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        for i, query in enumerate(self.test_queries, 1):
            self.run_test(query, i)
        
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze test results and provide summary"""
        print(f"\n{'='*80}")
        print("TEST RESULTS ANALYSIS")
        print(f"{'='*80}")
        
        total_tests = len(self.results)
        successful = sum(1 for r in self.results if r['success'])
        clarifications = sum(1 for r in self.results if r['needs_clarification'])
        errors = sum(1 for r in self.results if r['error'])
        exceptions = sum(1 for r in self.results if r.get('exception', False))
        
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful} ({successful/total_tests*100:.1f}%)")
        print(f"Clarifications: {clarifications} ({clarifications/total_tests*100:.1f}%)")
        print(f"Errors: {errors} ({errors/total_tests*100:.1f}%)")
        print(f"Exceptions: {exceptions} ({exceptions/total_tests*100:.1f}%)")
        
        # Show failed tests
        failed_tests = [r for r in self.results if not r['success'] and not r['needs_clarification']]
        if failed_tests:
            print(f"\nFAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   Test {test['test_number']}: {test['query']}")
                if test.get('error_message'):
                    print(f"      → {test['error_message']}")
        
        # Show clarification requests
        clarification_tests = [r for r in self.results if r['needs_clarification']]
        if clarification_tests:
            print(f"\nCLARIFICATION REQUESTS ({len(clarification_tests)}):")
            for test in clarification_tests:
                print(f"   Test {test['test_number']}: {test['query']}")
                print(f"      → {test['clarification_message']}")
        
        # Save detailed results to file
        self.save_results()
    
    def save_results(self):
        """Save detailed test results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "summary": {
                "successful": sum(1 for r in self.results if r['success']),
                "clarifications": sum(1 for r in self.results if r['needs_clarification']),
                "errors": sum(1 for r in self.results if r['error']),
                "exceptions": sum(1 for r in self.results if r.get('exception', False))
            },
            "test_results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {filename}")

def main():
    """Main entry point"""
    try:
        tester = QueryNERTester()
        tester.run_all_tests()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have set the GROQ_API_KEY environment variable:")
        print("  export GROQ_API_KEY='your-api-key-here'")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
