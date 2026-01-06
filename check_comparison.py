"""
So sanh thong so giua he thong va thoitiet360.edu.vn
"""
import pandas as pd
import sys

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("="*70)
print("SO SANH THONG SO GIUA HE THONG VA THOITIET360.EDU.VN")
print("="*70)

# Doc du lieu da crawl
df = pd.read_csv('thoitiet360_data.csv')

print("\n[1] THONG SO TU THOITIET360.EDU.VN:")
print("-" * 70)
for col in ['Temp', 'Pressure', 'Wind', 'Rain', 'Cloud', 'Gust']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        total = len(df)
        if non_null > 0:
            print(f"  [OK] {col}: {non_null}/{total} records co du lieu")
        else:
            print(f"  [NO] {col}: Khong co du lieu")
    else:
        print(f"  [NO] {col}: Khong co cot nay")

print("\n[2] HE THONG CUA BAN:")
print("-" * 70)
system_params = ['Temp', 'Rain', 'Cloud', 'Pressure', 'Wind', 'Gust']
print("  Cac thong so:", ', '.join(system_params))

print("\n[3] KET QUA SO SANH:")
print("-" * 70)

# Kiem tra tung thong so
matched = []
missing = []

for param in system_params:
    if param in df.columns:
        non_null = df[param].notna().sum()
        if non_null > 0:
            matched.append(param)
        else:
            missing.append(param)
    else:
        missing.append(param)

print(f"\n[KHOP] ({len(matched)}/6): {', '.join(matched)}")
print(f"[KHONG KHOP] ({len(missing)}/6): {', '.join(missing)}")

print("\n" + "="*70)
if len(matched) >= 4:
    print("KET LUAN: Co the tiep tuc crawl!")
    print(f"  - Co {len(matched)}/6 thong so khop")
    print("  - Co the so sanh: Temp, Rain, Pressure, Wind")
else:
    print("KET LUAN: Khong nen crawl")
    print(f"  - Chi co {len(matched)}/6 thong so khop")
print("="*70)

