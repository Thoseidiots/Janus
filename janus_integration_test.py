"""
Janus Integration Test - Verify All Components Work Together

Test script to verify all components are properly integrated
and can work together as a unified system.
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JanusIntegrationTest:
    """Integration test for Janus components"""
    
    def __init__(self):
        self.test_results = {}
        self.components = {}
        self.start_time = datetime.now()
        
        print("Janus Integration Test")
        print("=" * 40)
        print("Testing all components and integration")
        print()
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("STEP 1: COMPONENT AVAILABILITY TEST")
        print("-" * 40)
        
        # Test component imports
        components_tested = 0
        
        # Test finance system
        try:
            from standalone_finance import StandaloneFinance
            self.components["finance"] = StandaloneFinance()
            print("  Finance system: AVAILABLE")
            components_tested += 1
        except ImportError as e:
            print(f"  Finance system: NOT AVAILABLE - {e}")
        
        # Test AI brain
        try:
            from avus_brain import AvusBrain
            self.components["ai"] = AvusBrain()
            print("  AI brain: AVAILABLE")
            components_tested += 1
        except ImportError as e:
            print(f"  AI brain: NOT AVAILABLE - {e}")
        
        # Test browser automation
        try:
            from browser_automation import BrowserAutomationAgent
            self.components["browser"] = BrowserAutomationAgent()
            print("  Browser automation: AVAILABLE")
            components_tested += 1
        except ImportError as e:
            print(f"  Browser automation: NOT AVAILABLE - {e}")
        
        # Test QR scanner
        try:
            from janus_qr_scanner import JanusQRScanner
            self.components["qr"] = JanusQRScanner()
            print("  QR scanner: AVAILABLE")
            components_tested += 1
        except ImportError as e:
            print(f"  QR scanner: NOT AVAILABLE - {e}")
        
        # Test dual task manager
        try:
            from janus_dual_task_manager import JanusDualTaskManager
            self.components["task_manager"] = JanusDualTaskManager()
            print("  Task manager: AVAILABLE")
            components_tested += 1
        except ImportError as e:
            print(f"  Task manager: NOT AVAILABLE - {e}")
        
        print(f"Components available: {components_tested}/5")
        self.test_results["component_availability"] = components_tested >= 3
        
        print("\nSTEP 2: BASIC FUNCTIONALITY TEST")
        print("-" * 40)
        
        # Test finance system
        if "finance" in self.components:
            try:
                # Import the fixed finance system
                from finance_simple_fixed import TransactionType, PaymentMethod
                transaction = self.components["finance"].create_transaction(
                    transaction_type=TransactionType.INCOME,
                    amount=100.0,
                    currency="USD",
                    method=PaymentMethod.REVOLUT
                )
                print(f"  Finance test: PASSED (Transaction: {transaction.id})")
                self.test_results["finance_basic"] = True
            except Exception as e:
                print(f"  Finance test: FAILED - {e}")
                self.test_results["finance_basic"] = False
        
        # Test AI brain
        if "ai" in self.components:
            try:
                if self.components["ai"].ensure_loaded():
                    response = self.components["ai"].ask("Test response - say 'OK'", max_tokens=10)
                    if response and len(response.strip()) > 0:
                        print("  AI brain test: PASSED")
                        self.test_results["ai_basic"] = True
                    else:
                        print("  AI brain test: FAILED - Empty response")
                        self.test_results["ai_basic"] = False
                else:
                    print("  AI brain test: FAILED - Could not load")
                    self.test_results["ai_basic"] = False
            except Exception as e:
                print(f"  AI brain test: FAILED - {e}")
                self.test_results["ai_basic"] = False
        
        # Test browser automation
        if "browser" in self.components:
            try:
                if self.components["browser"].start_browser(headless=True):
                    self.components["browser"].navigate_to("https://www.google.com")
                    current_url = self.components["browser"].get_current_url()
                    if "google.com" in current_url:
                        print("  Browser automation test: PASSED")
                        self.test_results["browser_basic"] = True
                    else:
                        print("  Browser automation test: FAILED - Wrong URL")
                        self.test_results["browser_basic"] = False
                    self.components["browser"].driver.quit()
                else:
                    print("  Browser automation test: FAILED - Could not start")
                    self.test_results["browser_basic"] = False
            except Exception as e:
                print(f"  Browser automation test: FAILED - {e}")
                self.test_results["browser_basic"] = False
        
        print("\nSTEP 3: INTEGRATION TEST")
        print("-" * 30)
        
        # Test finance + AI integration
        if "finance" in self.components and "ai" in self.components:
            try:
                # Import the fixed finance system
                from finance_simple_fixed import TransactionType, PaymentMethod
                
                # Create a transaction for AI work
                transaction = self.components["finance"].create_transaction(
                    transaction_type=TransactionType.INCOME,
                    amount=150.0,
                    currency="USD",
                    method=PaymentMethod.REVOLUT,
                    description="AI content writing service"
                )
                
                # Generate AI content
                content = self.components["ai"].ask("Write a short business introduction", max_tokens=100)
                
                if len(content) > 10 and transaction.id:
                    print("  Finance + AI integration: PASSED")
                    self.test_results["finance_ai_integration"] = True
                else:
                    print("  Finance + AI integration: FAILED")
                    self.test_results["finance_ai_integration"] = False
            except Exception as e:
                print(f"  Finance + AI integration: FAILED - {e}")
                self.test_results["finance_ai_integration"] = False
        
        # Test browser + finance integration
        if "browser" in self.components and "finance" in self.components:
            try:
                # Import the fixed finance system
                from finance_simple_fixed import TransactionType, PaymentMethod
                
                # Start browser
                self.components["browser"].start_browser(headless=True)
                
                # Create transaction for browser work
                transaction = self.components["finance"].create_transaction(
                    transaction_type=TransactionType.INCOME,
                    amount=200.0,
                    currency="USD",
                    method=PaymentMethod.REVOLUT,
                    description="Browser automation service"
                )
                
                # Navigate to a site
                self.components["browser"].navigate_to("https://httpbin.org/get")
                page_source = self.components["browser"].get_page_source()
                
                if len(page_source) > 500 and transaction.id:
                    print("  Browser + Finance integration: PASSED")
                    self.test_results["browser_finance_integration"] = True
                else:
                    print("  Browser + Finance integration: FAILED")
                    self.test_results["browser_finance_integration"] = False
                
                self.components["browser"].driver.quit()
            except Exception as e:
                print(f"  Browser + Finance integration: FAILED - {e}")
                self.test_results["browser_finance_integration"] = False
        
        print("\nSTEP 4: PERFORMANCE TEST")
        print("-" * 30)
        
        # Test AI performance
        if "ai" in self.components:
            try:
                start_time = time.time()
                response = self.components["ai"].ask("Generate a 100-word business summary", max_tokens=150)
                end_time = time.time()
                
                generation_time = end_time - start_time
                word_count = len(response.split())
                
                print(f"  AI Performance: {word_count} words in {generation_time:.2f}s")
                print(f"  Speed: {word_count/generation_time:.1f} words/second")
                
                if word_count >= 80 and generation_time < 30:
                    print("  AI performance test: PASSED")
                    self.test_results["ai_performance"] = True
                else:
                    print("  AI performance test: FAILED")
                    self.test_results["ai_performance"] = False
            except Exception as e:
                print(f"  AI performance test: FAILED - {e}")
                self.test_results["ai_performance"] = False
        
        # Test finance performance
        if "finance" in self.components:
            try:
                # Import the fixed finance system
                from finance_simple_fixed import TransactionType, PaymentMethod
                
                start_time = time.time()
                
                # Create multiple transactions
                transactions = []
                for i in range(5):
                    txn = self.components["finance"].create_transaction(
                        transaction_type=TransactionType.INCOME,
                        amount=float(10 + i * 10),
                        currency="USD",
                        method=PaymentMethod.REVOLUT
                    )
                    transactions.append(txn)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"  Finance Performance: {len(transactions)} transactions in {processing_time:.3f}s")
                print(f"  Speed: {len(transactions)/processing_time:.1f} transactions/second")
                
                if len(transactions) == 5 and processing_time < 2.0:
                    print("  Finance performance test: PASSED")
                    self.test_results["finance_performance"] = True
                else:
                    print("  Finance performance test: FAILED")
                    self.test_results["finance_performance"] = False
            except Exception as e:
                print(f"  Finance performance test: FAILED - {e}")
                self.test_results["finance_performance"] = False
        
        print("\nSTEP 5: SYSTEM READINESS ASSESSMENT")
        print("-" * 40)
        
        # Calculate overall readiness
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        readiness_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {passed_tests}")
        print(f"Readiness Score: {readiness_score:.1f}%")
        print()
        
        # Component status
        print("COMPONENT STATUS:")
        for component, instance in self.components.items():
            status = "ACTIVE" if instance else "INACTIVE"
            print(f"  {component.title()}: {status}")
        print()
        
        # Test results
        print("TEST RESULTS:")
        for test_name, result in self.test_results.items():
            status = "PASSED" if result else "FAILED"
            print(f"  {test_name}: {status}")
        print()
        
        # Overall assessment
        if readiness_score >= 80:
            print("SYSTEM ASSESSMENT: PRODUCTION READY")
            print("All critical components operational")
            print("Ready for 24/7 autonomous operation")
        elif readiness_score >= 60:
            print("SYSTEM ASSESSMENT: OPERATIONAL WITH LIMITATIONS")
            print("Core functionality available")
            print("Some features may be limited")
        else:
            print("SYSTEM ASSESSMENT: NEEDS ATTENTION")
            print("Critical components missing or failing")
            print("Not ready for production")
        
        print()
        print("INTEGRATION TEST COMPLETE")
        
        # Save test results
        self.save_test_results()
        
        return readiness_score
    
    def save_test_results(self):
        """Save test results to file"""
        try:
            results = {
                "test_timestamp": datetime.now().isoformat(),
                "test_duration": (datetime.now() - self.start_time).total_seconds(),
                "components": list(self.components.keys()),
                "test_results": self.test_results,
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for result in self.test_results.values() if result),
                "readiness_score": (sum(1 for result in self.test_results.values() if result) / len(self.test_results)) * 100 if self.test_results else 0
            }
            
            with open("integration_test_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            print("Test results saved to: integration_test_results.json")
        except Exception as e:
            print(f"Failed to save test results: {e}")

# Main execution
def run_integration_test():
    """Run the integration test"""
    tester = JanusIntegrationTest()
    readiness_score = tester.run_all_tests()
    
    return readiness_score

if __name__ == "__main__":
    print("Janus Integration Test")
    print("Testing all components and integration")
    print()
    
    try:
        score = run_integration_test()
        print(f"\nFinal Readiness Score: {score:.1f}%")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest error: {e}")
