import os
import json
from pathlib import Path

def categorize_file(filepath: Path) -> str:
    path_str = str(filepath.as_posix()).lower()
    
    if '.git' in path_str or '__pycache__' in path_str or '.pytest_cache' in path_str:
        return 'ignore'
        
    if 'venv' in path_str or 'node_modules' in path_str or '.next' in path_str:
        return 'ignore'

    if path_str.endswith('.csv') and 'dataset' not in path_str and 'catalog' not in path_str:
        return 'legacy'
        
    if 'research/evonex/src' in path_str and path_str.endswith('.py'):
        return 'core'
        
    if 'docs/' in path_str or path_str.endswith('.md'):
        return 'documentation'
        
    if 'research/evonex/datasets' in path_str:
        return 'generated'
        
    if 'apps/' in path_str or 'services/' in path_str:
        return 'infrastructure'
        
    if path_str.endswith('.ipynb'):
        return 'legacy'
        
    return 'research'

def main():
    root_dir = Path(__file__).resolve().parents[1]
    manifest = {}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', 'venv', '.next']]
        
        for file in files:
            filepath = Path(root) / file
            rel_path = filepath.relative_to(root_dir).as_posix()
            
            category = categorize_file(filepath)
            if category != 'ignore':
                manifest[rel_path] = category
                
    with open(root_dir / 'repository_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Generated manifest with {len(manifest)} tracked files.")
    
    # Identify legacy files
    legacy_files = [k for k, v in manifest.items() if v == 'legacy']
    print(f"Found {len(legacy_files)} legacy/obsolete files.")
    for f in legacy_files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()
