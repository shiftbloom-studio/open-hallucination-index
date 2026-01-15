import requests
import json
import time
import sys
import os

API_URL = "http://ohi-api:8080/api"
API_KEY = os.environ.get("API_API_KEY", "")

# List of test cases designed to trigger specific MCP sources
TEST_CASES = [
    {
        "source": "Wiki/General",
        "text": "Douglas Adams was born in Cambridge and wrote The Hitchhiker's Guide to the Galaxy.",
        "keywords": ["wikidata", "wikipedia", "dbpedia", "mediawiki"]
    },
    {
        "source": "Academic (OpenAlex/Crossref)",
        "text": "The attention mechanism in deep learning was introduced in the paper 'Attention Is All You Need' by Vaswani et al.",
        "keywords": ["openalex", "crossref", "academic", "semantic_scholar"]
    },
    {
        "source": "Medical (PubMed/ClinicalTrials)",
        "text": "Metformin is the first-line medication for the treatment of type 2 diabetes.",
        "keywords": ["pubmed", "ncbi", "clinical_trials", "medical"]
    },
    {
        "source": "News (GDELT)",
        "text": "The 2024 Taiwan presidential election resulted in a victory for Lai Ching-te.",
        "keywords": ["gdelt", "news"]
    },
    {
        "source": "Security (OSV)",
        "text": "The Log4Shell vulnerability (CVE-2021-44228) enables remote code execution in Log4j.",
        "keywords": ["osv", "cve", "security", "vulnerabilities"]
    },
    {
        "source": "Economics (World Bank)",
        "text": "The GDP growth rate of India exceeded 7% in 2023.",
        "keywords": ["world_bank", "economic", "worldbank"]
    }
]

def run_test():
    print(f"Checking connection to {API_URL}...")
    try:
        requests.get(f"{API_URL}/health/live", timeout=5)
    except Exception as e:
        print(f"⚠️ Could not connect to API: {e}")
        # Proceed anyway, maybe /v1/verify works
    
    print("Wait for API/MCP warmup (2s)...")
    time.sleep(2)
    
    results = {}
    
    for test in TEST_CASES:
        print(f"\nExample - {test['source']}")
        print(f"Input: \"{test['text'][:60]}...\"")
        
        payload = {
            "text": test["text"],
            "strategy": "adaptive", 
            "target_sources": 6,
            "use_cache": False
        }
        
        try:
            start_time = time.time()
            headers = {"X-API-Key": API_KEY} if API_KEY else {}
            res = requests.post(f"{API_URL}/v1/verify", json=payload, headers=headers, timeout=90)
            duration = time.time() - start_time
            
            if res.status_code == 200:
                data = res.json()
                # Handle nested trust_score object
                score = data.get("overall_score")
                if score is None and "trust_score" in data:
                    score = data["trust_score"].get("overall")
                
                # Collect sources from evidence
                found_sources = set()
                evidence_count = 0
                
                # Check for 'claims' (new API) or 'claim_verifications' (old API or domain model)
                verifications = data.get("claims", []) or data.get("claim_verifications", [])
                
                # Debug print
                with open("/tmp/debug.json", "w") as f:
                    f.write(json.dumps(data, indent=2))
                
                for verification in verifications:
                    trace = verification.get("trace", {})
                    # If trace is missing, maybe it's flattened? 
                    # No, we expect trace to be passed.
                    
                    for ev in trace.get("supporting_evidence", []) + trace.get("refuting_evidence", []):
                        evidence_count += 1
                        # Check structured_data first
                        details = ev.get("structured_data", {})
                        orig = details.get("original_source")
                        if orig:
                            found_sources.add(str(orig).lower())
                        
                        # Also check the main source field
                        src = ev.get("source", "")
                        if src:
                            found_sources.add(str(src).lower())
                        
                        # Debug evidence
                        # print(f"DEBUG EV: {src} / {orig}")

                        # Check structured_data first
                        details = ev.get("structured_data", {})
                        orig = details.get("original_source")
                        if orig:
                            found_sources.add(str(orig).lower())
                        
                        # Also check the main source field
                        src = ev.get("source", "")
                        if src:
                            found_sources.add(str(src).lower())
                            
                print(f"  Result: {res.status_code} OK ({duration:.2f}s)")
                print(f"  Score:  {score}")
                print(f"  Sources Found: {list(found_sources)}")
                
                # Check if we hit the expected source
                hit = any(k in s for s in found_sources for k in test["keywords"])
                if hit:
                     print("  ✅ Target source verified.")
                else:
                     print("  ⚠️ Sources didn't match specific target keywords (might be lumped into 'knowledge_graph').")
                     
            else:
                print(f"  ❌ FAILED: Status {res.status_code}")
                # print(res.text)
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")

if __name__ == "__main__":
    run_test()
