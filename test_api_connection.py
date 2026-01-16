"""
Quick test script to verify OHI API connectivity and authentication.

Usage:
    python test_api_connection.py
"""

import asyncio
import os
import sys

import httpx


async def test_api_connection():
    """Test OHI API connection and authentication."""
    # Read configuration
    api_host = os.getenv("OHI_API_HOST", "localhost")
    api_port = os.getenv("OHI_API_PORT", "8080")
    api_key = os.getenv("API_API_KEY")
    
    base_url = f"http://{api_host}:{api_port}"
    
    print("=" * 70)
    print("OHI API Connection Test")
    print("=" * 70)
    print(f"API Base URL: {base_url}")
    print(f"API Key configured: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key length: {len(api_key)} characters")
        print(f"API Key preview: {api_key[:8]}...{api_key[-4:]}")
    print("=" * 70)
    print()
    
    # Test 1: Health check
    print("[1/3] Testing health endpoint...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                print(f"✓ Health check passed (status: {response.status_code})")
            else:
                print(f"✗ Health check failed (status: {response.status_code})")
                print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"✗ Health check error: {type(e).__name__}: {e}")
        return False
    
    print()
    
    # Test 2: Verify endpoint without auth
    print("[2/3] Testing verify endpoint without API key...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/v1/verify",
                json={"text": "Test claim", "strategy": "adaptive"},
                timeout=30.0,
            )
            print(f"  Status: {response.status_code}")
            if response.status_code == 401:
                print(f"✓ Auth required (as expected)")
            elif response.status_code == 200:
                print(f"✓ API accepts requests without auth (auth disabled)")
            else:
                print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"✗ Verify endpoint error: {type(e).__name__}: {e}")
    
    print()
    
    # Test 3: Verify endpoint with auth
    if api_key:
        print("[3/3] Testing verify endpoint with API key...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/api/v1/verify",
                    json={
                        "text": "Python was created in 1991.",
                        "strategy": "adaptive",
                        "use_cache": False,
                    },
                    headers={
                        "X-API-Key": api_key,
                        "X-Benchmark-Mode": "true",
                    },
                    timeout=30.0,
                )
                print(f"  Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    trust_score = data.get("trust_score", {})
                    if isinstance(trust_score, dict):
                        score = trust_score.get("overall", trust_score.get("score", 0))
                    else:
                        score = trust_score
                    print(f"✓ Verification successful!")
                    print(f"  Trust score: {score}")
                    print(f"  Claims count: {len(data.get('claims', []))}")
                elif response.status_code == 401:
                    print(f"✗ Authentication failed - API key is invalid")
                    print(f"  Response: {response.text[:200]}")
                else:
                    print(f"✗ Request failed (status: {response.status_code})")
                    print(f"  Response: {response.text[:500]}")
        except Exception as e:
            print(f"✗ Verify endpoint error: {type(e).__name__}: {e}")
    else:
        print("[3/3] Skipping authenticated test (no API key configured)")
    
    print()
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        asyncio.run(test_api_connection())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
