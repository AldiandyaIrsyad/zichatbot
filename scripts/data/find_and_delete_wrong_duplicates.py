import csv
import os
import collections

csv_file = '/home/aldiandyath/repo/skripsi_app/scrapers/data2.csv'
dataset_dir = '/home/aldiandyath/repo/skripsi_app/scrapers/dataset'

# Read rows
rows = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Group by base name
base_names = collections.defaultdict(list)

for row in rows:
    pdf_path = row.get('pdf_path')
    if not pdf_path:
        continue
    link = row.get('Link', '')
    uuid = link.split('/')[-1]
    
    # Strip _{uuid}.pdf
    suffix = f"_{uuid}.pdf"
    if pdf_path.endswith(suffix):
        base_name = pdf_path[:-len(suffix)]
    else:
        # Fallback if somehow different
        base_name = pdf_path
    
    base_names[base_name].append(row)

deleted_count = 0
affected_rows = 0

for base_name, grouped_rows in base_names.items():
    if len(grouped_rows) > 1:
        # This was a duplicated name originally
        affected_rows += len(grouped_rows)
        for row in grouped_rows:
            abs_path = os.path.join('/home/aldiandyath/repo/skripsi_app/scrapers', row['pdf_path'])
            if os.path.exists(abs_path):
                os.remove(abs_path)
                deleted_count += 1
                print(f"Deleted {row['pdf_path']}")

print(f"Affected rows (originally duplicates): {affected_rows}")
print(f"Deleted {deleted_count} mismatched files.")
