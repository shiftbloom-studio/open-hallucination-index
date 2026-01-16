# Benchmark Error Troubleshooting Guide

## Common Issue: High Error Rate ("current_errors" in display)

If you're seeing almost every fact check marked as an Exception during benchmark runs, follow this troubleshooting guide.

---

## Root Causes

1. **API Authentication Failure** - API key not configured or invalid
2. **Connection Pool Exhaustion** - Too many concurrent requests
3. **API Server Not Running** - OHI API is down or unreachable
4. **Network/Timeout Issues** - Slow responses causing timeouts

---

## Quick Diagnosis

### Step 1: Verify API Connection

Run the connection test script:

```bash
python test_api_connection.py
```

This will check:
- API server health
- Authentication status
- Basic verification functionality

### Step 2: Check Environment Variables

Ensure these are set correctly:

```bash
# Windows (Git Bash / PowerShell)
echo $API_API_KEY
echo $OHI_API_HOST
echo $OHI_API_PORT

# Verify they're exported
export -p | grep API
```

**Critical:** The benchmark looks for `API_API_KEY` (not `OHI_API_KEY`).

### Step 3: Verify API is Running

```bash
# Check if API is running
curl http://localhost:8080/health

# Should return: {"status":"healthy",...}
```

### Step 4: Check Benchmark Logs

Look for specific error patterns in the console output:

- **"Authentication failed (401)"** → API key issue
- **"Connection error"** → API not reachable
- **"Request timeout"** → API too slow or overloaded
- **"Connection pool exhausted"** → Too much concurrency

---

## Solutions

### Solution 1: Fix API Key Configuration

The benchmark config reads from `API_API_KEY` environment variable:

```bash
# Set API key (match what's in your API .env file)
export API_API_KEY="your-api-key-here"

# Verify it's set
echo $API_API_KEY
```

Then run benchmark again.

### Solution 2: Reduce Concurrency

Edit `src/benchmark/comparison_config.py`:

```python
# Execution Parameters
concurrency: int = 2  # Reduce from default 3
ohi_concurrency: int = 2
```

Or set via environment:

```bash
export BENCHMARK_CONCURRENCY=2
export OHI_CONCURRENCY=2
```

### Solution 3: Increase Timeouts

If your local setup is slow:

```bash
export BENCHMARK_TIMEOUT=180.0  # Increase from 120s
```

### Solution 4: Check API Server Resources

If API is running but slow:

```bash
# Check Docker containers
docker stats

# Look for high CPU/memory usage on:
# - ohi-api
# - vllm
# - neo4j
# - qdrant
```

Consider:
- Reducing vLLM batch size
- Allocating more memory to containers
- Using a GPU for vLLM

### Solution 5: Run Smaller Benchmark

Test with fewer samples first:

```bash
# Small test run
python -m benchmark --hallucination-samples 10 --metrics hallucination

# If successful, gradually increase
python -m benchmark --hallucination-samples 30
```

---

## Recent Fixes Applied

The following improvements were made to fix persistent error issues:

1. **Enhanced Connection Pooling**
   - Increased max connections to 10 (was dynamic based on concurrency)
   - Added keep-alive with 30s expiry
   - Separate timeouts for connect/read/write/pool

2. **Retry Logic with Backoff**
   - Up to 3 retries on connection errors
   - Exponential backoff (0.5s, 1s, 2s)
   - Specific handling for `ConnectError`, `PoolTimeout`, `ConnectTimeout`

3. **Better Error Classification**
   - Distinguish timeout vs connection vs HTTP errors
   - Special handling for 401 (auth) and 422/403 (validation)
   - Detailed logging with error types

4. **API Key Logging**
   - Logs at startup if API key is configured
   - Shows key length for verification
   - Warns if missing

5. **Thread-Safe Client Management**
   - Added async lock for client creation
   - Proper cleanup with `close()` method
   - Prevents race conditions

6. **Reduced Default Concurrency**
   - Changed from 5 to 3 concurrent requests
   - Prevents overwhelming local API setup
   - Can still override via config/env

---

## Verification Steps After Fixes

1. **Run connection test:**
   ```bash
   python test_api_connection.py
   ```

2. **Check logs during benchmark:**
   - Look for "Creating HTTP client" messages
   - Should see "API key configured" if key is set
   - Watch for specific error types

3. **Monitor error rate:**
   - Errors should be < 5% for healthy runs
   - If still high, reduce concurrency further

4. **Check API logs:**
   ```bash
   docker logs ohi-api --tail=100
   ```
   - Look for 401s (auth issues)
   - Look for 500s (server errors)

---

## Environment Variable Reference

```bash
# API Configuration
export API_API_KEY="your-key-here"           # CRITICAL for auth
export OHI_API_HOST="localhost"              # Default: localhost
export OHI_API_PORT="8080"                   # Default: 8080
export OHI_STRATEGY="adaptive"               # Default: adaptive

# Benchmark Configuration
export BENCHMARK_CONCURRENCY=3               # Concurrent requests
export OHI_CONCURRENCY=3                     # OHI-specific concurrency
export BENCHMARK_TIMEOUT=120.0               # Request timeout (seconds)
export BENCHMARK_WARMUP=5                    # Warmup requests

# Dataset Configuration
export BENCHMARK_DATASET="path/to/dataset.csv"
export BENCHMARK_EVALUATORS="ohi,gpt4"       # Which evaluators to run
export BENCHMARK_METRICS="hallucination,latency"

# Optional: Redis (for cache testing)
export REDIS_HOST="localhost"
export REDIS_PORT=6379
export REDIS_PASSWORD=""  # If needed
```

---

## Still Having Issues?

1. **Check this file for updates** - Solutions may be added as new issues are discovered

2. **Enable verbose logging:**
   ```bash
   python -m benchmark --verbose
   ```

3. **Check API server logs:**
   ```bash
   docker logs ohi-api -f
   ```

4. **Verify all services are healthy:**
   ```bash
   docker ps
   curl http://localhost:8080/health/ready
   ```

5. **Test single verification:**
   ```python
   import httpx
   import asyncio
   
   async def test():
       async with httpx.AsyncClient() as client:
           response = await client.post(
               "http://localhost:8080/api/v1/verify",
               json={"text": "Test", "strategy": "adaptive"},
               headers={"X-API-Key": "your-key"},
           )
           print(response.status_code, response.json())
   
   asyncio.run(test())
   ```

---

## Summary Checklist

- [ ] API server is running (`curl http://localhost:8080/health`)
- [ ] `API_API_KEY` environment variable is set and matches API config
- [ ] Run `python test_api_connection.py` - all tests pass
- [ ] Concurrency is set appropriately (2-3 for local, higher for production)
- [ ] Timeouts are sufficient for your setup
- [ ] Docker containers have adequate resources
- [ ] Logs show successful HTTP client creation
- [ ] Error rate is < 5% during benchmark runs
