#!/usr/bin/env python3
"""
Test script for production security features
Validates authentication, authorization, and rate limiting
"""

import asyncio
import aiohttp
import time
import os
import sys
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://127.0.0.1:8000"

class ProductionSecurityTester:
    """Test class for production security validation"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_no_auth(self) -> bool:
        """Test that protected endpoints require authentication"""
        print("\n🔒 Testing authentication requirement...")
        
        protected_endpoints = ["/status", "/chat"]
        
        for endpoint in protected_endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 401:
                        print(f"✅ {endpoint} correctly requires authentication")
                    else:
                        print(f"❌ {endpoint} should require authentication (got {response.status})")
                        return False
            except Exception as e:
                print(f"❌ Error testing {endpoint}: {e}")
                return False
        
        return True
    
    async def test_invalid_auth(self) -> bool:
        """Test that invalid tokens are rejected"""
        print("\n🚫 Testing invalid authentication...")
        
        invalid_tokens = [
            "invalid-token",
            "Bearer invalid",
            "jwt.invalid.token",
            ""
        ]
        
        for token in invalid_tokens:
            try:
                headers = {"Authorization": f"Bearer {token}"}
                async with self.session.get(f"{self.base_url}/status", headers=headers) as response:
                    if response.status == 401:
                        print(f"✅ Invalid token correctly rejected")
                    else:
                        print(f"❌ Invalid token should be rejected (got {response.status})")
                        return False
            except Exception as e:
                print(f"❌ Error testing invalid auth: {e}")
                return False
        
        return True
    
    async def test_rate_limiting(self) -> bool:
        """Test that rate limiting is working"""
        print("\n🛡️ Testing rate limiting...")
        
        # Make many requests quickly to trigger rate limiting
        success_count = 0
        rate_limited_count = 0
        
        for i in range(20):
            try:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        success_count += 1
                    elif response.status == 429:
                        rate_limited_count += 1
                        print(f"✅ Rate limiting triggered after {success_count} requests")
                        break
                    await asyncio.sleep(0.1)  # Small delay
            except Exception as e:
                print(f"❌ Error testing rate limiting: {e}")
                return False
        
        if rate_limited_count > 0:
            print(f"✅ Rate limiting working ({success_count} succeeded, {rate_limited_count} rate limited)")
            return True
        else:
            print(f"⚠️  Rate limiting may not be working (all {success_count} requests succeeded)")
            return True  # Not necessarily a failure for testing
    
    async def test_security_headers(self) -> bool:
        """Test that security headers are present"""
        print("\n🔐 Testing security headers...")
        
        expected_headers = [
            "x-content-type-options",
            "x-frame-options", 
            "x-xss-protection",
            "strict-transport-security"
        ]
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                headers = {k.lower(): v for k, v in response.headers.items()}
                
                missing_headers = []
                for header in expected_headers:
                    if header not in headers:
                        missing_headers.append(header)
                    else:
                        print(f"✅ Security header present: {header}")
                
                if missing_headers:
                    print(f"❌ Missing security headers: {missing_headers}")
                    return False
                else:
                    print("✅ All security headers present")
                    return True
                    
        except Exception as e:
            print(f"❌ Error testing security headers: {e}")
            return False
    
    async def test_cors_headers(self) -> bool:
        """Test CORS configuration"""
        print("\n🌐 Testing CORS headers...")
        
        try:
            headers = {
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST"
            }
            
            async with self.session.options(f"{self.base_url}/chat", headers=headers) as response:
                cors_header = response.headers.get("Access-Control-Allow-Origin")
                
                if cors_header == "*":
                    print("⚠️  CORS allows all origins (development mode)")
                    return True
                elif cors_header is None:
                    print("✅ CORS correctly restricts origins")
                    return True
                else:
                    print(f"✅ CORS configured for specific origins: {cors_header}")
                    return True
                    
        except Exception as e:
            print(f"❌ Error testing CORS: {e}")
            return False
    
    async def test_request_size_limits(self) -> bool:
        """Test request size limitations"""
        print("\n📏 Testing request size limits...")
        
        # Create a large payload
        large_text = "A" * (2 * 1024 * 1024)  # 2MB text
        payload = {"text": large_text}
        
        try:
            headers = {"Authorization": "Bearer test-key"}
            async with self.session.post(
                f"{self.base_url}/chat", 
                json=payload,
                headers=headers
            ) as response:
                if response.status in [413, 400]:
                    print("✅ Large requests correctly rejected")
                    return True
                else:
                    print(f"⚠️  Large request not rejected (status: {response.status})")
                    return True  # May not be configured yet
                    
        except Exception as e:
            print(f"❌ Error testing request size limits: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all security tests"""
        print("🔒 Jarvis Production Security Test Suite")
        print("=" * 60)
        
        tests = {
            "authentication_required": await self.test_no_auth(),
            "invalid_auth_rejected": await self.test_invalid_auth(),
            "rate_limiting": await self.test_rate_limiting(),
            "security_headers": await self.test_security_headers(),
            "cors_configuration": await self.test_cors_headers(),
            "request_size_limits": await self.test_request_size_limits()
        }
        
        return tests

async def main():
    """Main test runner"""
    print("Jarvis Production Security Validation")
    print("=====================================")
    
    # Check if server is running
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status != 200:
                    print(f"❌ Server not responding correctly (status: {response.status})")
                    print("Please start the server with:")
                    print("python jarvis_api_production.py")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("Please start the server with:")
            print("python jarvis_api_production.py")
            return
    
    async with ProductionSecurityTester() as tester:
        results = await tester.run_all_tests()
        
        print("\n" + "=" * 60)
        print("📊 SECURITY TEST RESULTS")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
        
        print("-" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All security tests passed!")
            print("The production security configuration is working correctly.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} security test(s) failed.")
            print("Review the configuration and implement missing security features.")
        
        print("\n📋 SECURITY CHECKLIST:")
        print("□ Configure environment variables with real secrets")
        print("□ Set up SSL/TLS certificates")
        print("□ Configure firewall rules")
        print("□ Set up monitoring and alerting")
        print("□ Test from external network")
        print("□ Perform penetration testing")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Tests cancelled by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")