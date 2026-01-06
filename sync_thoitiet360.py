"""
Script để tự động đồng bộ files thoitiet360 từ GitHub
Chạy script này định kỳ (mỗi 10 phút) hoặc khi khởi động máy
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Đang kiểm tra cập nhật thoitiet360 từ GitHub...")
    print(f"  Thư mục làm việc: {SCRIPT_DIR}")
    
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
            print(f"  ⚠️  Không thể fetch từ GitHub (có thể không phải git repo): {result.stderr[:100]}")
            return False
        
        # Kiểm tra xem có commit mới không
        result = subprocess.run(
            ['git', 'rev-list', '--count', 'HEAD..origin/main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Không thể kiểm tra commits: {result.stderr[:100]}")
            return False
        
        commits_behind = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        if commits_behind > 0:
            print(f"  📥 Tìm thấy {commits_behind} commit(s) mới. Đang pull cập nhật...")
            
            # Pull các thay đổi
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            
            if result.returncode == 0:
                print(f"  ✅ Đã pull thành công {commits_behind} commit(s)")
                print(f"  📄 Các file đã cập nhật: thoitiet360_data.csv, database, và các file khác")
                
                # Cập nhật database nếu có file CSV mới
                try:
                    from database import init_database, get_db_connection
                    import pandas as pd
                    
                    csv_file = os.path.join(SCRIPT_DIR, 'thoitiet360_data.csv')
                    if os.path.exists(csv_file):
                        print(f"  💾 Đang cập nhật database từ CSV...")
                        df = pd.read_csv(csv_file)
                        from crawl_thoitiet360 import preprocess_thoitiet360_data, save_to_database
                        df_processed = preprocess_thoitiet360_data(df)
                        save_to_database(df_processed)
                except Exception as e:
                    print(f"  ⚠️  Không thể cập nhật database: {str(e)[:100]}")
                
                return True
            else:
                print(f"  ⚠️  Lỗi khi pull: {result.stderr[:100]}")
                return False
        else:
            print("  ✅ Đã cập nhật mới nhất, không có thay đổi")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_and_pull()

