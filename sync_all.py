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
                
                # Lấy danh sách các file đã thay đổi sau khi pull
                result = subprocess.run(
                    ['git', 'diff', '--name-only', 'HEAD@{1}', 'HEAD'],
                    capture_output=True,
                    text=True,
                    cwd=SCRIPT_DIR
                )
                
                changed_files = []
                if result.returncode == 0 and result.stdout.strip():
                    changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                
                # Nếu không lấy được từ diff, thử cách khác
                if not changed_files:
                    result = subprocess.run(
                        ['git', 'log', '--name-only', '--pretty=format:', '-1'],
                        capture_output=True,
                        text=True,
                        cwd=SCRIPT_DIR
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip() and not f.startswith('commit')]
                
                # Kiểm tra các file quan trọng
                important_files = [
                    'weather.db',
                    'weather_models_final.pkl',
                    'weather_models_improved.pkl',
                    'weather_all_cities.csv',
                    'thoitiet360_data.csv'
                ]
                
                print("  📄 Các file đã được cập nhật:")
                updated_files = []
                for file in important_files:
                    file_path = os.path.join(SCRIPT_DIR, file)
                    if os.path.exists(file_path):
                        # Kiểm tra xem file có trong danh sách thay đổi không
                        is_changed = any(file in changed_file or changed_file.endswith(file) for changed_file in changed_files)
                        if is_changed or commits_behind > 0:  # Nếu có commit mới, có thể file đã được cập nhật
                            updated_files.append(file)
                            status = "🔄" if is_changed else "✓"
                            print(f"    {status} {file}")
                
                # Hiển thị tất cả các file đã thay đổi
                if changed_files:
                    print()
                    print(f"  📋 Tổng cộng {len(changed_files)} file đã thay đổi:")
                    for file in changed_files[:20]:  # Chỉ hiển thị 20 file đầu
                        print(f"    • {file}")
                    if len(changed_files) > 20:
                        print(f"    ... và {len(changed_files) - 20} file khác")
                
                # Ghi log vào file
                log_file = os.path.join(SCRIPT_DIR, 'sync_log.txt')
                try:
                    # Đảm bảo file có BOM nếu là file mới
                    file_exists = os.path.exists(log_file)
                    with open(log_file, 'a', encoding='utf-8-sig') as f:
                        if not file_exists:
                            f.write('\ufeff')  # UTF-8 BOM
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        f.write(f"\n[{timestamp}] Pull thành công - {commits_behind} commit(s)\n")
                        f.write(f"Files updated: {', '.join(updated_files) if updated_files else 'None'}\n")
                        if changed_files:
                            f.write(f"All changed files ({len(changed_files)}): {', '.join(changed_files[:10])}\n")
                except Exception as e:
                    print(f"  ⚠️  Không thể ghi log: {str(e)[:50]}")
                
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
            
            # Ghi log ngay cả khi không có thay đổi
            log_file = os.path.join(SCRIPT_DIR, 'sync_log.txt')
            try:
                # Đảm bảo file có BOM nếu là file mới
                file_exists = os.path.exists(log_file)
                with open(log_file, 'a', encoding='utf-8-sig') as f:
                    if not file_exists:
                        f.write('\ufeff')  # UTF-8 BOM
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{timestamp}] Đã kiểm tra - Không có cập nhật mới\n")
            except Exception as e:
                print(f"  ⚠️  Không thể ghi log: {str(e)[:50]}")
            
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

