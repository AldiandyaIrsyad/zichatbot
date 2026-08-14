import csv
import os
import re
import time
import shutil
from urllib.parse import urljoin
from collections import Counter
from patchright.sync_api import sync_playwright

CSV_INPUT = "data2.csv"
CSV_OUTPUT = "data3.csv"
DATASET_DIR = "dataset"
PROFILE_DIR = ".browser_profile"
MAX_TITLE_LENGTH = 150
DOWNLOAD_LINK_RE = re.compile(r'(?:https?://[^"\'\s<>]+?)?reviewDokumenPeraturan/download/[^"\'\s<>]+\.pdf')

def get_uuid_from_link(link):
    # Example: https://jdih.upi.edu/content/list_detail/a179c33b-6647-42e1-9aac-2bdee1d82799
    parts = link.rstrip('/').split('/')
    return parts[-1] if parts else ""

def sanitize_filename(title, uuid_suffix=""):
    # Remove invalid characters for filenames
    filename = re.sub(r'[\\/*?:"<>|]', "", title)
    # Collapse all whitespace (including newlines) into single underscores
    filename = re.sub(r'\s+', "_", filename.strip())
    
    # We'll append uuid_suffix to make it unique
    if uuid_suffix:
        suffix = f"_{uuid_suffix}"
    else:
        suffix = ""
        
    max_len = MAX_TITLE_LENGTH - len(suffix)
    if len(filename) > max_len:
        filename = filename[:max_len].strip("_")
        
    return filename + suffix + ".pdf"

def find_download_url(page):
    match = DOWNLOAD_LINK_RE.search(page.content())
    return match.group(0) if match else None

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    rows = []
    with open(CSV_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    # Find duplicate pdf_paths
    paths = [r.get('pdf_path', '') for r in rows if r.get('pdf_path', '')]
    counts = Counter(paths)
    duplicate_paths = set([p for p, count in counts.items() if count > 1])

    temp_csv = CSV_OUTPUT + ".tmp"
    with open(temp_csv, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        # Check how many are missing or duplicates
        to_download = []
        for row in rows:
            pdf_path = row.get('pdf_path', '')
            needs_download = False
            
            if not pdf_path:
                needs_download = True
            elif not os.path.exists(pdf_path):
                needs_download = True
            elif pdf_path in duplicate_paths:
                needs_download = True
                
            if needs_download:
                to_download.append(row)

        print(f"Found {len(to_download)} PDFs to download (missing or duplicates).")

        if to_download:
            with sync_playwright() as p:
                print("Launching browser with persistent profile...")
                context = p.chromium.launch_persistent_context(
                    PROFILE_DIR,
                    headless=False,
                    accept_downloads=True,
                    no_viewport=True,
                )
                page = context.pages[0] if context.pages else context.new_page()

                for index, row in enumerate(rows, 1):
                    pdf_path = row.get('pdf_path', '')
                    needs_download = False
                    
                    if not pdf_path or not os.path.exists(pdf_path):
                        needs_download = True
                    elif pdf_path in duplicate_paths:
                        needs_download = True

                    if not needs_download:
                        writer.writerow(row)
                        f_out.flush()
                        continue

                    link = row.get('Link', '')
                    title = row.get('Title', f"Document_{index}")
                    uuid_suffix = get_uuid_from_link(link)
                    print(f"\n[{index}/{len(rows)}] Processing missing/duplicate PDF: {link}")

                    try:
                        page.goto(link, wait_until='domcontentloaded')
                        download_url = None
                        max_retries = 15

                        for attempt in range(max_retries):
                            page.wait_for_timeout(3000)
                            download_url = find_download_url(page)
                            if download_url:
                                break
                            else:
                                print(f"    [Attempt {attempt+1}/{max_retries}] Download link not found. Waiting...")

                        if download_url:
                            if not download_url.startswith('http'):
                                download_url = '/' + download_url.lstrip('/')
                            download_url = urljoin(page.url, download_url)
                            print(f"    Found download URL: {download_url}")

                            # Generate a unique filename by appending the UUID from the link
                            filename = sanitize_filename(title, uuid_suffix=uuid_suffix)
                            filepath = os.path.join(DATASET_DIR, filename)

                            response = page.request.get(download_url)
                            if response.status == 200:
                                with open(filepath, 'wb') as pdf_file:
                                    pdf_file.write(response.body())
                                print(f"    Successfully downloaded to {filepath}")
                                row['pdf_path'] = filepath
                            else:
                                print(f"    Failed to download, status code: {response.status}")
                        else:
                            print("    Could not find download URL on the page after waiting.")

                    except Exception as e:
                        print(f"    Error processing {link}: {e}")

                    writer.writerow(row)
                    f_out.flush()
                    time.sleep(3)

                context.close()
        else:
            for row in rows:
                writer.writerow(row)

    shutil.move(temp_csv, CSV_OUTPUT)
    print(f"Finished. Saved updated data to {CSV_OUTPUT}")

if __name__ == '__main__':
    main()
