
"""
System Testing Script
Tests all modules and API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:5000"

class SystemTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = []
    
    def test_endpoint(self, name, method, url, data=None, expected_status=200):
        """Test single endpoint"""
        print(f"\n🧪 Testing: {name}")
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            
            status = response.status_code
            success = status == expected_status
            
            result = {
                "test": name,
                "status": status,
                "expected": expected_status,
                "success": success,
                "time": response.elapsed.total_seconds()
            }
            
            self.test_results.append(result)
            
            if success:
                print(f"   ✅ PASS ({status}) - {response.elapsed.total_seconds():.2f}s")
            else:
                print(f"   ❌ FAIL (Expected {expected_status}, got {status})")
            
            return response
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            self.test_results.append({
                "test": name,
                "success": False,
                "error": str(e)
            })
            return None
    
    def run_all_tests(self):
        """Run comprehensive tests"""
        print("\n" + "="*60)
        print("🚀 STARTING SYSTEM TESTS")
        print("="*60)
        
        # Test 1: Homepage
        self.test_endpoint(
            "Homepage",
            "GET",
            f"{BASE_URL}/"
        )
        
        # Test 2: Login page
        self.test_endpoint(
            "Login Page",
            "GET",
            f"{BASE_URL}/login"
        )
        
        # Test 3: Register page
        self.test_endpoint(
            "Register Page",
            "GET",
            f"{BASE_URL}/register"
        )
        
        # Test 4: API - Technical questions
        self.test_endpoint(
            "Get Technical Question",
            "POST",
            f"{BASE_URL}/api/technical/start",
            data={"company_id": 1},
            expected_status=401  # Not authenticated
        )
        
        # Test 5: API - GD topics
        self.test_endpoint(
            "Get GD Topic",
            "POST",
            f"{BASE_URL}/api/gd/start",
            data={"company_id": 1},
            expected_status=401
        )
        
        # Test 6: API - Report generation
        self.test_endpoint(
            "Generate Report",
            "GET",
            f"{BASE_URL}/api/report/generate",
            expected_status=401
        )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.get("success"))
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        
        if failed > 0:
            print("\n⚠️  Failed Tests:")
            for r in self.test_results:
                if not r.get("success"):
                    print(f"   - {r['test']}")
        
        print("="*60)

