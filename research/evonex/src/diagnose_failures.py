import os
import time
import pandas as pd
import lightkurve as lk
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose():
    catalog_csv = os.path.join(os.path.dirname(__file__), "..", "datasets", "toi_catalog.csv")
    df = pd.read_csv(catalog_csv)
    
    # Take a random 50 targets for diagnosis
    sample = df.sample(n=50, random_state=42)
    
    results = []
    
    for idx, row in sample.iterrows():
        tic_id = str(row['tic_id'])
        target_name = f"TIC {tic_id}"
        
        entry = {
            "tic": tic_id,
            "search_success": False,
            "spoc_found": False,
            "available_authors": [],
            "available_missions": [],
            "error": None
        }
        
        try:
            # Broad search across all missions/authors
            search_result = lk.search_lightcurve(target_name)
            
            if len(search_result) > 0:
                entry["search_success"] = True
                
                # Check what authors are available
                authors = set(search_result.author)
                missions = set(search_result.mission)
                
                entry["available_authors"] = list(authors)
                entry["available_missions"] = list(missions)
                
                if "SPOC" in authors:
                    entry["spoc_found"] = True
                    
        except Exception as e:
            entry["error"] = str(e)
            
        results.append(entry)
        time.sleep(0.5) # Gentle rate limit
        
    # Summarize
    total = len(results)
    no_results = sum(1 for r in results if not r["search_success"] and r["error"] is None)
    spoc_count = sum(1 for r in results if r["spoc_found"])
    errors = sum(1 for r in results if r["error"] is not None)
    
    author_dist = {}
    for r in results:
        for a in r["available_authors"]:
            author_dist[a] = author_dist.get(a, 0) + 1
            
    summary = {
        "total_checked": total,
        "total_with_any_lc": total - no_results - errors,
        "total_with_spoc": spoc_count,
        "total_no_results": no_results,
        "total_errors": errors,
        "author_distribution": author_dist
    }
    
    print("\n--- DIAGNOSTIC SUMMARY ---")
    print(json.dumps(summary, indent=4))
    
    # Show a few specific failures
    print("\n--- SAMPLE FAILURES ---")
    failures = [r for r in results if not r["spoc_found"]]
    for f in failures[:5]:
        print(f)

if __name__ == "__main__":
    diagnose()
