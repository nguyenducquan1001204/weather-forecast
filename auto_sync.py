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

def check_and_pull():
    """Check for updates and pull if available"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for updates...")
    
    try:
        # Fetch latest changes
        result = subprocess.run(
            ['git', 'fetch', 'origin', 'main'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Check if there are new commits
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD..origin/main'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        commits_behind = int(result.stdout.strip())
        
        if commits_behind > 0:
            print(f"  Found {commits_behind} new commit(s). Pulling updates...")
            
            # Pull changes
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
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
        return False

if __name__ == "__main__":
    check_and_pull()

