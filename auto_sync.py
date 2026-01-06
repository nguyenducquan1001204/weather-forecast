"""
Script để tự động đồng bộ files từ GitHub khi workflow hoàn thành
Chạy script này định kỳ (ví dụ: mỗi 5-10 phút) hoặc như một scheduled task
"""
import subprocess
import sys
import os
from datetime import datetime

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Lấy thư mục nơi script này được đặt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_and_pull():
    """Kiểm tra cập nhật và pull nếu có"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for updates...")
    print(f"  Working directory: {SCRIPT_DIR}")
    
    try:
        # Chuyển đến thư mục script để đảm bảo đúng vị trí
        os.chdir(SCRIPT_DIR)
        
        # Lấy các thay đổi mới nhất
        result = subprocess.run(
            ['git', 'fetch', 'origin', 'main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ❌ Lỗi khi fetch: {result.stderr}")
            return False
        
        # Kiểm tra xem có commit mới không
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD..origin/main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ❌ Lỗi khi kiểm tra commits: {result.stderr}")
            return False
        
        commits_behind = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        if commits_behind > 0:
            print(f"  Tìm thấy {commits_behind} commit(s) mới. Đang pull cập nhật...")
            
            # Pull các thay đổi
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            
            if result.returncode == 0:
                print(f"  ✅ Đã pull thành công {commits_behind} commit(s)")
                print(f"  Các file đã cập nhật: weather.db, weather_models_final.pkl, weather_all_cities.csv")
                return True
            else:
                print(f"  ❌ Lỗi khi pull: {result.stderr}")
                return False
        else:
            print("  ✅ Đã cập nhật mới nhất")
            return False
            
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_and_pull()

