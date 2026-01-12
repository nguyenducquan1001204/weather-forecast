# -*- coding: utf-8 -*-
import os
import sys

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

log_file = os.path.join(os.path.dirname(__file__), 'sync_log.txt')

print("="*70)
print("XEM LOG DONG BO TU GITHUB")
print("="*70)
print()

if os.path.exists(log_file):
    try:
        with open(log_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
        
        if lines:
            print(f"Tong so dong trong log: {len(lines)}")
            print()
            print("50 dong cuoi cung:")
            print("-"*70)
            for line in lines[-50:]:
                print(line.rstrip())
        else:
            print("File log trong.")
    except Exception as e:
        print(f"Loi khi doc file log: {e}")
else:
    print("Chua co file log. Chua co lan dong bo nao duoc thuc hien.")
    print()
    print("Ban co the chay thu cong: python sync_all.py")

print()
print("="*70)

