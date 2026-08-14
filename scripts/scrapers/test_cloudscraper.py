import cloudscraper
from bs4 import BeautifulSoup
import re

url = "https://jdih.upi.edu/content/list_detail/a2451374-a6eb-4b8b-9430-6f5fca12a6cd"
scraper = cloudscraper.create_scraper()
response = scraper.get(url)

print("Status Code:", response.status_code)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    for link in links:
        if 'download' in link['href'] or '.pdf' in link['href'] or 'reviewDokumenPeraturan' in link['href']:
            print("Found download link:", link['href'])
else:
    print("Failed to fetch.")
