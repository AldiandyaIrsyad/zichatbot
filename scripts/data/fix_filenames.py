import csv
import os

csv_file = '/home/aldiandyath/repo/skripsi_app/scrapers/data2.csv'
dataset_dir = '/home/aldiandyath/repo/skripsi_app/scrapers/dataset'

# Read the CSV
rows = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

renamed_count = 0
missing_old_files = 0

for row in rows:
    old_pdf_path = row.get('pdf_path')
    if not old_pdf_path:
        continue
        
    # Extract UUID from Link
    link = row.get('Link', '')
    uuid = link.split('/')[-1]
    
    # Check if the old_pdf_path already contains the uuid (just in case)
    if uuid in old_pdf_path:
        continue
        
    # Generate new pdf_path
    # old_pdf_path format: "dataset/Title_Truncated.pdf"
    if old_pdf_path.endswith('.pdf'):
        base_path = old_pdf_path[:-4]
    else:
        base_path = old_pdf_path
        
    new_pdf_path = f"{base_path}_{uuid}.pdf"
    
    # Rename physical file if exists
    # The actual file path on disk:
    # Notice that old_pdf_path is something like "dataset/..."
    # We must construct absolute paths
    abs_old_path = os.path.join('/home/aldiandyath/repo/skripsi_app/scrapers', old_pdf_path)
    abs_new_path = os.path.join('/home/aldiandyath/repo/skripsi_app/scrapers', new_pdf_path)
    
    if os.path.exists(abs_old_path):
        os.rename(abs_old_path, abs_new_path)
        renamed_count += 1
    else:
        # File doesn't exist, maybe it was overwritten or already renamed by a duplicate
        if not os.path.exists(abs_new_path):
            missing_old_files += 1
            
    # Update row
    row['pdf_path'] = new_pdf_path

print(f"Renamed {renamed_count} files.")
print(f"Files not found (already renamed or overwritten): {missing_old_files}")

# Write updated rows back to CSV
with open(csv_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    
print("Updated data2.csv successfully.")
