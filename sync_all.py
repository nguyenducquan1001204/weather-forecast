import subprocess
import sys
import os
from datetime import datetime

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_and_pull():
    """
    Kiểm tra và pull tất cả các file từ GitHub, sau đó import dữ liệu vào database nếu cần
    """
    print("="*70)
    print("KIỂM TRA VÀ ĐỒNG BỘ DỮ LIỆU TỪ GITHUB")
    print("="*70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Đang kiểm tra cập nhật...")
    print(f"  Thư mục làm việc: {SCRIPT_DIR}")
    print()
    
    try:
        os.chdir(SCRIPT_DIR)
        
        # Fetch từ GitHub
        result = subprocess.run(
            ['git', 'fetch', 'origin', 'main'],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR
        )
        
        if result.returncode != 0:
            print(f"  ⚠️  Không thể fetch từ GitHub (có thể không phải git repo): {result.stderr[:100]}")
            return False
        
        # Kiểm tra số commits mới
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
            print()
            
            # Pull từ GitHub
            result = subprocess.run(
                ['git', 'pull', 'origin', 'main'],
                capture_output=True,
                text=True,
                cwd=SCRIPT_DIR
            )
            
            if result.returncode == 0:
                print(f"  ✅ Đã pull thành công {commits_behind} commit(s)")
                print()
                print("  📄 Các file đã được cập nhật:")
                
                # Kiểm tra các file quan trọng
                important_files = [
                    'weather.db',
                    'weather_models_final.pkl',
                    'weather_models_improved.pkl',
                    'weather_all_cities.csv',
                    'thoitiet360_data.csv'
                ]
                
                updated_files = []
                for file in important_files:
                    file_path = os.path.join(SCRIPT_DIR, file)
                    if os.path.exists(file_path):
                        updated_files.append(file)
                        print(f"    ✓ {file}")
                
                print()
                
                # Import thoitiet360_data.csv vào database nếu có
                csv_file = os.path.join(SCRIPT_DIR, 'thoitiet360_data.csv')
                if os.path.exists(csv_file):
                    try:
                        print("  💾 Đang import dữ liệu thoitiet360 từ CSV vào database...")
                        import import_thoitiet360_to_db
                        count = import_thoitiet360_to_db.import_csv_to_database()
                        if count > 0:
                            print(f"  ✅ Đã import {count} records vào database")
                        else:
                            print("  ℹ️  Không có dữ liệu mới để import")
                    except Exception as e:
                        print(f"  ⚠️  Không thể import vào database: {str(e)[:100]}")
                
                print()
                print("="*70)
                print("✅ HOÀN TẤT ĐỒNG BỘ")
                print("="*70)
                return True
            else:
                print(f"  ⚠️  Lỗi khi pull: {result.stderr[:100]}")
                return False
        else:
            print("  ✅ Đã cập nhật mới nhất, không có thay đổi")
            print()
            print("="*70)
            return False
            
    except Exception as e:
        print(f"  ⚠️  Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_and_pull()

