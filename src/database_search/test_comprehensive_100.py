#!/usr/bin/env python3
"""
Comprehensive test script for Ocean Query NER system
Tests 100 diverse queries including edge cases, temporal variations, and failure scenarios
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path so we can import queryNER
sys.path.append(str(Path(__file__).parent))

from queryNER import OceanQueryExtractorGroq

class Comprehensive100Tester:
    """Comprehensive test suite with 100 diverse queries"""
    
    def __init__(self):
        # Initialize both extractors
        print("Initializing extractors...")
        self.extractor_llm_only = OceanQueryExtractorGroq(use_spacy_preprocessing=False)
        self.extractor_hybrid = OceanQueryExtractorGroq(use_spacy_preprocessing=True)
        
        # 100 comprehensive test queries
        self.test_queries = [
            # Basic temporal queries (1-10)
            "What is the temperature in Cambridge Bay today?",
            "Show me salinity data for Saanich Inlet yesterday",
            "How cold was it in Cambridge Bay last Monday?",
            "Get oxygen levels for Strait of Georgia tomorrow",
            "What's the pressure in Cambridge Bay right now?",
            "Show me pH data for Saanich Inlet this morning",
            "Temperature readings for Cambridge Bay tonight",
            "Salinity levels in Strait of Georgia this afternoon",
            "How warm is it in Cambridge Bay at noon?",
            "Get turbidity data for Saanich Inlet at midnight",
            
            # Month-level queries (11-20)
            "What are the salinity levels in Cambridge Bay for the month of January?",
            "Show me temperature data for Saanich Inlet in February",
            "Get oxygen levels for the entire month of March in Cambridge Bay",
            "pH readings for Strait of Georgia during April",
            "Turbidity data for Cambridge Bay throughout May",
            "Show me conductivity for Saanich Inlet in June 2024",
            "Temperature trends for Cambridge Bay in July",
            "Salinity patterns for Strait of Georgia in August",
            "Chlorophyll levels for Saanich Inlet during September",
            "Pressure data for Cambridge Bay for the whole month of October",
            
            # Specific date and time queries (21-30)
            "What was the temperature at Cambridge Bay on January 15th at 2 PM?",
            "Show me salinity at noon on Tuesday in Saanich Inlet",
            "Get pressure data for Cambridge Bay tomorrow morning",
            "Temperature on December 25th at 6 AM in Strait of Georgia",
            "Oxygen levels on March 3rd at 10:30 PM in Cambridge Bay",
            "pH data for Saanich Inlet on Friday at 4 PM",
            "Salinity on January 1st at midnight in Cambridge Bay",
            "Turbidity readings on August 15th at 3:45 PM in Strait of Georgia",
            "Temperature at 11:15 AM on Thursday in Saanich Inlet",
            "Conductivity on September 30th at 7:22 PM in Cambridge Bay",
            
            # Date range queries (31-40)
            "Temperature data for Cambridge Bay from January 1 to January 7",
            "Show me salinity trends for the past week in Saanich Inlet",
            "Get oxygen levels from last Monday to this Friday",
            "pH data for Cambridge Bay between March 10 and March 20",
            "Turbidity readings from February 1st through February 14th in Strait of Georgia",
            "Temperature range for Saanich Inlet from December 2024 to January 2025",
            "Salinity data for Cambridge Bay over the last 30 days",
            "Conductivity trends for Strait of Georgia from summer to fall",
            "Oxygen levels for Saanich Inlet from Monday to Wednesday",
            "Pressure variations in Cambridge Bay during the first week of May",
            
            # Complex temporal expressions (41-50)
            "Temperature in Cambridge Bay on the first Monday of January",
            "Salinity data for Saanich Inlet last Tuesday evening",
            "What was the temperature last week on Wednesday?",
            "Oxygen levels in Strait of Georgia next Friday afternoon",
            "pH readings for Cambridge Bay two days ago",
            "Turbidity in Saanich Inlet three weeks from now",
            "Temperature on the third Thursday of February in Cambridge Bay",
            "Salinity during the second weekend of March in Strait of Georgia",
            "Conductivity on the last day of the month in Saanich Inlet",
            "Pressure readings during the middle of next week in Cambridge Bay",
            
            # Parameter variations and synonyms (51-60)
            "How salty is the water in Cambridge Bay right now?",
            "What's the dissolved oxygen content in Saanich Inlet today?",
            "Show me pH levels for Cambridge Bay yesterday",
            "Get turbidity readings for Saanich Inlet this morning",
            "How hot is the water in Strait of Georgia?",
            "Salt content in Cambridge Bay last week",
            "Water temperature in Saanich Inlet at noon",
            "Dissolved O2 levels in Strait of Georgia yesterday",
            "Acidity measurements for Cambridge Bay today",
            "Water clarity in Saanich Inlet this afternoon",
            
            # Location variations (61-70)
            "Temperature in Strait of Georgia today",
            "Salinity levels in Iqaluktuuttiaq last week",
            "Oxygen data for Cambridge Bay, Nunavut yesterday",
            "pH readings in Saanich Inlet, BC this morning",
            "Turbidity in the Strait of Georgia tomorrow",
            "Temperature at Iqaluktuuttiaq yesterday evening",
            "Salinity in Cambridge Bay, NU last Monday",
            "Conductivity for Saanich Inlet on Vancouver Island",
            "Pressure data for Strait of Georgia, BC",
            "Chlorophyll in Cambridge Bay Arctic waters",
            
            # Depth-specific queries (71-80)
            "Temperature at 10 meters depth in Cambridge Bay today",
            "Salinity at 50m in Saanich Inlet yesterday",
            "Oxygen levels at 100 meter depth in Strait of Georgia",
            "pH at 25 meters deep in Cambridge Bay last week",
            "Turbidity at 5m depth in Saanich Inlet this morning",
            "Temperature at surface level in Cambridge Bay",
            "Salinity at 200 meters down in Strait of Georgia",
            "Conductivity at 75m depth in Saanich Inlet",
            "Pressure at 150 meters deep in Cambridge Bay",
            "Chlorophyll at 30m depth in Strait of Georgia yesterday",
            
            # Ambiguous temporal queries (81-85)
            "Temperature data for Cambridge Bay",
            "Show me recent salinity in Saanich Inlet",
            "Get current oxygen levels for Strait of Georgia",
            "pH readings for Cambridge Bay sometime last month",
            "Turbidity data for Saanich Inlet around noon",
            
            # Edge cases and potential failures (86-100)
            "What's the weather like in Cambridge Bay?",
            "Temperature data for New York yesterday",
            "Show me data for Cambridge Bay",
            "Get temperature",
            "How deep is Cambridge Bay?",
            "What time is it in Saanich Inlet?",
            "Show me fish populations in Strait of Georgia",
            "Get ice thickness for Cambridge Bay",
            "What's the tide level in Saanich Inlet?",
            "Show me wind speed for Cambridge Bay today",
            "Get wave height data for Strait of Georgia",
            "What's the current direction in Saanich Inlet?",
            "Show me sediment data for Cambridge Bay",
            "Get plankton counts for Strait of Georgia",
            "What's the visibility in Saanich Inlet today?"
        ]
        
        self.results = []
        self.categories = {
            "basic_temporal": list(range(1, 11)),
            "month_level": list(range(11, 21)),
            "specific_datetime": list(range(21, 31)),
            "date_ranges": list(range(31, 41)),
            "complex_temporal": list(range(41, 51)),
            "parameter_variations": list(range(51, 61)),
            "location_variations": list(range(61, 71)),
            "depth_queries": list(range(71, 81)),
            "ambiguous_temporal": list(range(81, 86)),
            "edge_cases": list(range(86, 101))
        }
    
    def run_comparison_test(self, query, test_number):
        """Run comparison test for a single query"""
        print(f"\n{'='*80}")
        print(f"TEST {test_number}: {query}")
        print(f"{'='*80}")
        
        # Test LLM-only approach
        print("\n🔹 LLM-ONLY APPROACH:")
        try:
            result_llm = self.extractor_llm_only.extract_parameters(query)
            self.print_result(result_llm, "LLM-only")
        except Exception as e:
            print(f"💥 LLM-only EXCEPTION: {str(e)}")
            result_llm = {"status": "exception", "error": str(e)}
        
        # Test hybrid approach
        print("\n🔹 HYBRID (LLM + spaCy) APPROACH:")
        try:
            result_hybrid = self.extractor_hybrid.extract_parameters(query)
            self.print_result(result_hybrid, "Hybrid")
        except Exception as e:
            print(f"💥 Hybrid EXCEPTION: {str(e)}")
            result_hybrid = {"status": "exception", "error": str(e)}
        
        # Compare results
        comparison = self.compare_results(result_llm, result_hybrid, query)
        
        # Categorize the test
        category = self.get_test_category(test_number)
        
        test_result = {
            "test_number": test_number,
            "query": query,
            "category": category,
            "llm_only": result_llm,
            "hybrid": result_hybrid,
            "comparison": comparison
        }
        
        self.results.append(test_result)
        return test_result
    
    def get_test_category(self, test_number):
        """Get the category for a test number"""
        for category, numbers in self.categories.items():
            if test_number in numbers:
                return category
        return "unknown"
    
    def print_result(self, result, approach_name):
        """Print result in a consistent format"""
        if result.get("status") == "success":
            params = result["parameters"]
            print(f"✅ {approach_name} SUCCESS")
            print(f"   Parameter: {params['instrument_type']}")
            print(f"   Location: {params['location']}")
            print(f"   Time Range: {params['start_time']} to {params['end_time']}")
            print(f"   Duration: {params['time_range_hours']:.1f} hours")
            if params.get('depth_meters'):
                print(f"   Depth: {params['depth_meters']} meters")
            
            # Show spaCy info if available
            metadata = result.get("metadata", {})
            if metadata.get("spacy_preprocessing") and metadata.get("spacy_entities"):
                spacy_ents = metadata["spacy_entities"]
                print(f"   spaCy found: {len(spacy_ents.get('dates', []))} dates, {len(spacy_ents.get('temporal_patterns', []))} patterns")
                
        elif result.get("status") == "error":
            print(f"❌ {approach_name} ERROR: {result.get('message', 'Unknown error')}")
        else:
            print(f"⚠️ {approach_name} STATUS: {result.get('status', 'No status')}")
    
    def compare_results(self, llm_result, hybrid_result, query):
        """Compare the two results and provide analysis"""
        comparison = {
            "both_successful": False,
            "improvement": "none",
            "temporal_extraction_different": False,
            "parameter_extraction_different": False,
            "location_extraction_different": False,
            "notes": []
        }
        
        llm_success = llm_result.get("status") == "success"
        hybrid_success = hybrid_result.get("status") == "success"
        
        comparison["both_successful"] = llm_success and hybrid_success
        
        if not llm_success and hybrid_success:
            comparison["improvement"] = "hybrid_better"
            comparison["notes"].append("Hybrid succeeded where LLM-only failed")
        elif llm_success and not hybrid_success:
            comparison["improvement"] = "llm_better"
            comparison["notes"].append("LLM-only succeeded where hybrid failed")
        elif llm_success and hybrid_success:
            # Both succeeded, compare quality
            llm_params = llm_result.get("parameters", {})
            hybrid_params = hybrid_result.get("parameters", {})
            
            # Compare temporal extraction
            if llm_params.get("start_time") != hybrid_params.get("start_time"):
                comparison["temporal_extraction_different"] = True
                comparison["notes"].append("Different temporal extraction results")
            
            # Compare parameter extraction
            if llm_params.get("instrument_type") != hybrid_params.get("instrument_type"):
                comparison["parameter_extraction_different"] = True
                comparison["notes"].append("Different parameter extraction")
            
            # Compare location extraction
            if llm_params.get("location") != hybrid_params.get("location"):
                comparison["location_extraction_different"] = True
                comparison["notes"].append("Different location extraction")
            
            # Check if hybrid used spaCy preprocessing successfully
            hybrid_meta = hybrid_result.get("metadata", {})
            if hybrid_meta.get("spacy_preprocessing") and hybrid_meta.get("spacy_entities"):
                spacy_ents = hybrid_meta["spacy_entities"]
                if spacy_ents.get("dates") or spacy_ents.get("temporal_patterns"):
                    comparison["notes"].append("spaCy provided temporal preprocessing")
        
        # Show comparison briefly
        if comparison["improvement"] != "none":
            print(f"\n📊 COMPARISON: {comparison['improvement']}")
        
        return comparison
    
    def run_all_tests(self):
        """Run all 100 comparison tests"""
        print("🧪 STARTING COMPREHENSIVE 100-QUERY TEST SUITE")
        print(f"Running {len(self.test_queries)} test queries...")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        for i, query in enumerate(self.test_queries, 1):
            self.run_comparison_test(query, i)
        
        self.analyze_comprehensive_results()
    
    def analyze_comprehensive_results(self):
        """Analyze comprehensive results by category and overall"""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        total_tests = len(self.results)
        llm_successes = sum(1 for r in self.results if r['llm_only'].get('status') == 'success')
        hybrid_successes = sum(1 for r in self.results if r['hybrid'].get('status') == 'success')
        
        hybrid_better = sum(1 for r in self.results if r['comparison']['improvement'] == 'hybrid_better')
        llm_better = sum(1 for r in self.results if r['comparison']['improvement'] == 'llm_better')
        temporal_differences = sum(1 for r in self.results if r['comparison']['temporal_extraction_different'])
        parameter_differences = sum(1 for r in self.results if r['comparison']['parameter_extraction_different'])
        location_differences = sum(1 for r in self.results if r['comparison']['location_extraction_different'])
        
        print(f"OVERALL RESULTS:")
        print(f"Total Tests: {total_tests}")
        print(f"LLM-only Success Rate: {llm_successes}/{total_tests} ({llm_successes/total_tests*100:.1f}%)")
        print(f"Hybrid Success Rate: {hybrid_successes}/{total_tests} ({hybrid_successes/total_tests*100:.1f}%)")
        print(f"")
        print(f"Hybrid Better: {hybrid_better} tests")
        print(f"LLM-only Better: {llm_better} tests")
        print(f"Temporal Differences: {temporal_differences} tests")
        print(f"Parameter Differences: {parameter_differences} tests")
        print(f"Location Differences: {location_differences} tests")
        
        # Analyze by category
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, test_numbers in self.categories.items():
            category_results = [r for r in self.results if r['test_number'] in test_numbers]
            if category_results:
                cat_llm_success = sum(1 for r in category_results if r['llm_only'].get('status') == 'success')
                cat_hybrid_success = sum(1 for r in category_results if r['hybrid'].get('status') == 'success')
                cat_total = len(category_results)
                
                print(f"  {category.replace('_', ' ').title()}: LLM {cat_llm_success}/{cat_total} ({cat_llm_success/cat_total*100:.1f}%) | Hybrid {cat_hybrid_success}/{cat_total} ({cat_hybrid_success/cat_total*100:.1f}%)")
        
        # Show failure patterns
        llm_failures = [r for r in self.results if r['llm_only'].get('status') != 'success']
        hybrid_failures = [r for r in self.results if r['hybrid'].get('status') != 'success']
        
        if llm_failures:
            print(f"\n❌ LLM-ONLY FAILURES ({len(llm_failures)}):")
            failure_categories = {}
            for failure in llm_failures:
                category = failure['category']
                if category not in failure_categories:
                    failure_categories[category] = []
                failure_categories[category].append(failure)
            
            for category, failures in failure_categories.items():
                print(f"  {category.replace('_', ' ').title()}: {len(failures)} failures")
                for f in failures[:3]:  # Show first 3 examples
                    print(f"    • Test {f['test_number']}: {f['query'][:60]}...")
        
        if hybrid_failures:
            print(f"\n❌ HYBRID FAILURES ({len(hybrid_failures)}):")
            failure_categories = {}
            for failure in hybrid_failures:
                category = failure['category']
                if category not in failure_categories:
                    failure_categories[category] = []
                failure_categories[category].append(failure)
            
            for category, failures in failure_categories.items():
                print(f"  {category.replace('_', ' ').title()}: {len(failures)} failures")
                for f in failures[:3]:  # Show first 3 examples
                    print(f"    • Test {f['test_number']}: {f['query'][:60]}...")
        
        # Show improvements
        if hybrid_better > 0:
            print(f"\n✨ HYBRID IMPROVEMENTS ({hybrid_better}):")
            improvements = [r for r in self.results if r['comparison']['improvement'] == 'hybrid_better']
            for imp in improvements:
                print(f"   Test {imp['test_number']}: {imp['query']}")
        
        if llm_better > 0:
            print(f"\n⚡ LLM-ONLY ADVANTAGES ({llm_better}):")
            advantages = [r for r in self.results if r['comparison']['improvement'] == 'llm_better']
            for adv in advantages:
                print(f"   Test {adv['test_number']}: {adv['query']}")
        
        # Save detailed results
        self.save_comprehensive_results()
    
    def save_comprehensive_results(self):
        """Save detailed comprehensive results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_100_results_{timestamp}.json"
        
        # Calculate category statistics
        category_stats = {}
        for category, test_numbers in self.categories.items():
            category_results = [r for r in self.results if r['test_number'] in test_numbers]
            if category_results:
                llm_success = sum(1 for r in category_results if r['llm_only'].get('status') == 'success')
                hybrid_success = sum(1 for r in category_results if r['hybrid'].get('status') == 'success')
                total = len(category_results)
                
                category_stats[category] = {
                    "total_tests": total,
                    "llm_successes": llm_success,
                    "hybrid_successes": hybrid_success,
                    "llm_success_rate": llm_success / total * 100,
                    "hybrid_success_rate": hybrid_success / total * 100
                }
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "summary": {
                "llm_successes": sum(1 for r in self.results if r['llm_only'].get('status') == 'success'),
                "hybrid_successes": sum(1 for r in self.results if r['hybrid'].get('status') == 'success'),
                "hybrid_better": sum(1 for r in self.results if r['comparison']['improvement'] == 'hybrid_better'),
                "llm_better": sum(1 for r in self.results if r['comparison']['improvement'] == 'llm_better'),
                "temporal_differences": sum(1 for r in self.results if r['comparison']['temporal_extraction_different']),
                "parameter_differences": sum(1 for r in self.results if r['comparison']['parameter_extraction_different']),
                "location_differences": sum(1 for r in self.results if r['comparison']['location_extraction_different'])
            },
            "category_stats": category_stats,
            "test_results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive results saved to: {filename}")

def main():
    """Main entry point"""
    try:
        tester = Comprehensive100Tester()
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