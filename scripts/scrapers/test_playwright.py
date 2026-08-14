from playwright.sync_api import sync_playwright

url = "https://jdih.upi.edu/content/list_detail/a2451374-a6eb-4b8b-9430-6f5fca12a6cd"

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')
        
        # Wait a bit just in case Cloudflare takes a few seconds
        page.wait_for_timeout(3000)
        
        links = page.locator('a').all()
        for link in links:
            href = link.get_attribute('href')
            if href and ('download' in href or '.pdf' in href or 'reviewDokumenPeraturan' in href):
                print("Found download link:", href)
                
        browser.close()

if __name__ == '__main__':
    run()
