
import requests
import json
import time
import sys
import os
from typing import List, Dict, Any

API_BASE = "http://ohi-api:8080"
API_KEY = os.environ.get("API_API_KEY", "")
SKIP_WORLD_BANK = os.environ.get("SKIP_WORLD_BANK", "false").lower() in {"1", "true", "yes"}

# Distinct queries designed to trigger specific tools or domains where those tools are primary
TOOL_TESTS = [
    {
        "tool_match": "search_wikipedia",
        "name": "Wikipedia",
        "query": "Douglas Adams was born in Cambridge.",
    },
    {
        "tool_match": "search_wikidata",
        "name": "Wikidata",
        "query": "The capital of France is Paris.",
    },
    {
        "tool_match": "search_dbpedia",
        "name": "DBpedia",
        "query": "Berlin is a city in Germany.",
    },
    {
        "tool_match": "search_openalex",
        "name": "OpenAlex",
        "query": "The paper 'Attention Is All You Need' introduced the Transformer model.",
    },
    {
        "tool_match": "search_crossref",
        "name": "Crossref",
        "query": "The DOI 10.1038/nature12345 refers to a specific scientific publication.",
    },
    {
        "tool_match": "search_pubmed",
        "name": "PubMed",
        "query": "Metformin is used to treat type 2 diabetes.",
    },
    {
        "tool_match": "search_europepmc",
        "name": "EuropePMC",
        "query": "Recent studies on malaria prevention in Europe.",
    },
    {
        "tool_match": "search_clinical_trials",
        "name": "ClinicalTrials",
        "query": "NCT04368728 is a clinical trial identifier for a COVID-19 vaccine study.",
    },
    {
        "tool_match": "search_gdelt",
        "name": "GDELT",
        "query": "The 2024 Taiwan presidential election resulted in a victory for Lai Ching-te.",
    },
    {
        "tool_match": "search_vulnerabilities",
        "name": "OSV (Security)",
        "query": "The Log4Shell vulnerability CVE-2021-44228 affects Log4j library.",
    },
    {
        "tool_match": "query-docs",
        "name": "Context7",
        "query": "Next.js App Router uses Server Components by default.",
    },
    {
        "tool_match": "get_world_bank_indicator",
        "name": "World Bank",
        "query": "The GDP growth of Brazil in 2022 was positive.",
    }
]

def check_health():
    print(f"Checking API health at {API_BASE}/health/live...")
    try:
        res = requests.get(f"{API_BASE}/health/live", timeout=5)
        if res.status_code == 200:
            print("✅ API is healthy.")
            return True
        else:
            print(f"❌ API returned status {res.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def run_individual_tests():
    if not check_health():
        return

    print(f"\n🚀 Starting Individual MCP Tool Tests ({len(TOOL_TESTS)} tools)")
    print("Each test sends a query tailored to trigger specific MCP sources.")
    
    results = []

    for test in TOOL_TESTS:
        tool_match = test["tool_match"]
        name = test["name"]
        query = test["query"]

        if SKIP_WORLD_BANK and tool_match == "get_world_bank_indicator":
            print("Skipping World Bank test (SKIP_WORLD_BANK enabled).")
            continue
        
        print(f"\n----------------------------------------------------------------")
        print(f"Testing Tool: {name} (expected: {tool_match})")
        print(f"Query: \"{query}\"")
        
        payload = {
            "text": query,
            "strategy": "mcp_enhanced",  # FORCE MCP usage
            "target_sources": 10,
            "use_cache": False
        }
        
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        
        try:
            start_ts = time.time()
            # User requested timeouts to avoid infinite hangs
            res = requests.post(f"{API_BASE}/api/v1/verify", json=payload, headers=headers, timeout=30)
            duration = time.time() - start_ts
            
            if res.status_code != 200:
                print(f"❌ Request failed with {res.status_code}: {res.text}")
                results.append({"name": name, "status": "FAIL", "msg": f"HTTP {res.status_code}"})
                continue
                
            data = res.json()
            
            # Inspect trace for the specific tool
            found = False
            found_evidence_count = 0
            
            verifications = data.get("claims", []) or data.get("claim_verifications", [])
            all_sources_found = set()
            
            for v in verifications:
                trace = v.get("trace", {})
                
                # Helper to check evidence
                def check_ev(ev_list):
                    nonlocal found, found_evidence_count
                    for ev in ev_list:
                        source = ev.get("source", "")
                        # Check structured data for tool name
                        struct = ev.get("structured_data", {})
                        tool_name = struct.get("tool", "")
                        
                        all_sources_found.add(f"{source}:{tool_name}")
                        
                        if tool_match in tool_name:
                            found = True
                            found_evidence_count += 1

                check_ev(trace.get("supporting_evidence", []))
                check_ev(trace.get("refuting_evidence", []))

            if found:
                print(f"✅ SUCCESS: Found evidence via tool '{tool_match}' ({found_evidence_count} items)")
                results.append({"name": name, "status": "PASS", "duration": f"{duration:.2f}s", "sources": list(all_sources_found)})
            else:
                print(f"⚠️ WARNING: Did not find evidence from tool '{tool_match}'")
                print(f"   Sources found: {list(all_sources_found)}")
                results.append({"name": name, "status": "FAIL", "duration": f"{duration:.2f}s", "sources": list(all_sources_found)})

        except requests.exceptions.Timeout:
            print(f"❌ TIMEOUT: Request timed out after 30 seconds.")
            results.append({"name": name, "status": "TIMEOUT", "msg": "Timeout (30s)"})
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({"name": name, "status": "ERROR", "msg": str(e)})

    print("\n\n📊 TEST SUMMARY")
    print("="*60)
    print(f"{'Tool':<20} | {'Status':<10} | {'Time':<8} | {'Details'}")
    print("-" * 60)
    for r in results:
        status_icon = "✅" if r["status"] == "PASS" else ("⚠️" if r["status"] == "PARTIAL" else "❌")
        details = f"Sources: {len(r.get('sources', []))}"
        if r["status"] != "PASS":
            details = r.get("msg", "")
        print(f"{status_icon} {r['name']:<18} | {r['status']:<10} | {r.get('duration', 'N/A'):<8} | {details}")
    print("="*60)

if __name__ == "__main__":
    run_individual_tests()
