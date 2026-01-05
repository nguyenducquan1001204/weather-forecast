"""
Script to automatically sync files from GitHub when workflow completes
Run this script periodically (e.g., every 5-10 minutes) or as a scheduled task
"""
import subprocess
import sys
import os
from datetime import datetime

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_and_pull():
    """Check for updates and pull if available"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for updates...")
    print(f"  Working directory: {SCRIPT_DIR}")
    
    try:
        # Change to script directory to ensure we're in the right place
        os.chdir(SCRIPT_DIR)
        
        # Fetch latest changes
        result = subprocess.run(
            ['git', 'fetch', 'origin', 'main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ❌ Error fetching: {result.stderr}")
            return False
        
        # Check if there are new commits
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD..origin/main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ❌ Error checking commits: {result.stderr}")
            return False
        
        commits_behind = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        if commits_behind > 0:
            print(f"  Found {commits_behind} new commit(s). Pulling updates...")
            
            # Pull changes
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            
            if result.returncode == 0:
                print(f"  ✅ Successfully pulled {commits_behind} commit(s)")
                print(f"  Updated files: weather.db, weather_models_final.pkl, weather_all_cities.csv")
                return True
            else:
                print(f"  ❌ Error pulling: {result.stderr}")
                return False
        else:
            print("  ✅ Already up to date")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_and_pull()

