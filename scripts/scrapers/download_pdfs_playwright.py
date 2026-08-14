import csv
import os
import re
import time
from urllib.parse import urljoin
from patchright.sync_api import sync_playwright

CSV_INPUT = "data.csv"
CSV_OUTPUT = "data2.csv"
DATASET_DIR = "dataset"
PROFILE_DIR = ".browser_profile"
MAX_TITLE_LENGTH = 150
DOWNLOAD_LINK_RE = re.compile(r'(?:https?://[^"\'\s<>]+?)?reviewDokumenPeraturan/download/[^"\'\s<>]+\.pdf')

def sanitize_filename(title):
    # Remove invalid characters for filenames
    filename = re.sub(r'[\\/*?:"<>|]', "", title)
    # Collapse all whitespace (including newlines) into single underscores
    filename = re.sub(r'\s+', "_", filename.strip())
    # Truncate to MAX_TITLE_LENGTH
    if len(filename) > MAX_TITLE_LENGTH:
        filename = filename[:MAX_TITLE_LENGTH].strip("_")
    return filename + ".pdf"

def load_completed_links():
    """Return the set of Link values that already have a non-empty pdf_path in CSV_OUTPUT."""
    completed = set()
    if os.path.exists(CSV_OUTPUT):
        with open(CSV_OUTPUT, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('pdf_path'):
                    completed.add(row.get('Link', ''))
    return completed

def find_download_url(page):
    match = DOWNLOAD_LINK_RE.search(page.content())
    return match.group(0) if match else None

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    rows = []

    # Read the original CSV
    with open(CSV_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['pdf_path'] if reader.fieldnames else []
        for row in reader:
            rows.append(row)

    completed_links = load_completed_links()
    if completed_links:
        print(f"Resuming: {len(completed_links)} link(s) already downloaded, will be skipped.")

    # We will write to the output CSV dynamically so we don't lose progress if it crashes
    with open(CSV_OUTPUT, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with sync_playwright() as p:
            # patchright patches the CDP leaks Cloudflare fingerprints, so the managed
            # challenge clears on its own here (confirmed: no stealth layer needed).
            # headless=False + a persistent profile also means any challenge that does
            # require human input only needs solving once; the cookie is then reused
            # (from disk) on every future run.
            print("Launching browser with persistent profile...")
            context = p.chromium.launch_persistent_context(
                PROFILE_DIR,
                headless=False,
                accept_downloads=True,
                no_viewport=True,
            )
            page = context.pages[0] if context.pages else context.new_page()

            for index, row in enumerate(rows, 1):
                link = row.get('Link', '')
                title = row.get('Title', f"Document_{index}")

                if link in completed_links:
                    print(f"\n[{index}/{len(rows)}] Skipping (already downloaded): {link}")
                    writer.writerow(row)
                    f_out.flush()
                    continue

                print(f"\n[{index}/{len(rows)}] Processing: {link}")

                try:
                    # Navigate to the detail page
                    page.goto(link, wait_until='domcontentloaded')

                    download_url = None
                    max_retries = 15 # Wait up to 45 seconds total for Cloudflare bypass

                    for attempt in range(max_retries):
                        page.wait_for_timeout(3000) # Wait 3 seconds per attempt

                        download_url = find_download_url(page)

                        if download_url:
                            break # Found it!
                        else:
                            print(f"    [Attempt {attempt+1}/{max_retries}] Download link not found. Waiting for Cloudflare bypass...")

                    if download_url:
                        # The href is normally already an absolute URL; if a root-relative
                        # path slipped through, resolve it against the domain root (not the
                        # current detail-page path) so urljoin doesn't mangle it.
                        if not download_url.startswith('http'):
                            download_url = '/' + download_url.lstrip('/')
                        download_url = urljoin(page.url, download_url)
                        print(f"    Found download URL: {download_url}")

                        filename = sanitize_filename(title)
                        filepath = os.path.join(DATASET_DIR, filename)

                        # We use page.request.get to download the file using the browser's context (cookies, etc)
                        response = page.request.get(download_url)
                        if response.status == 200:
                            with open(filepath, 'wb') as pdf_file:
                                pdf_file.write(response.body())
                            print(f"    Successfully downloaded to {filepath}")
                            row['pdf_path'] = filepath
                        else:
                            print(f"    Failed to download, status code: {response.status}")
                            row['pdf_path'] = ""
                    else:
                        print("    Could not find download URL on the page after waiting.")
                        row['pdf_path'] = ""

                except Exception as e:
                    print(f"    Error processing {link}: {e}")
                    row['pdf_path'] = ""

                # Write row and flush to disk
                writer.writerow(row)
                f_out.flush()

                # Sleep to avoid rate limits
                time.sleep(3)

            context.close()
            print("Finished processing all links.")

if __name__ == '__main__':
    main()
