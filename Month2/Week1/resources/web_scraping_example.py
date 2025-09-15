"""
Web Scraping Example for Housing Data

This script demonstrates how to scrape housing data from a real estate website
and prepare it for linear regression analysis.

Note: Always check a website's terms of service and robots.txt before scraping.
This example is for educational purposes only.
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import random

class RealEstateScraper:
    """A simple web scraper for real estate data."""
    
    def __init__(self, base_url, headers=None):
        """Initialize the scraper with base URL and headers."""
        self.base_url = base_url
        self.headers = headers or {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.data = []
    
    def get_page(self, page_number):
        """Fetch a single page of listings."""
        url = f"{self.base_url}/pg-{page_number}_p/"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching page {page_number}: {e}")
            return None
    
    def parse_listing(self, listing):
        """Extract data from a single listing."""
        try:
            # These selectors are example placeholders
            # You'll need to inspect the actual website and update them
            price = listing.select_one('.list-card-price').text.strip()
            address = listing.select_one('.list-card-addr').text.strip()
            details = listing.select('.list-card-details')
            
            # Clean and parse the data
            price = int(price.replace('$', '').replace(',', ''))
            
            # Extract features (this is just an example)
            beds = 0
            baths = 0
            sqft = 0
            
            for detail in details:
                text = detail.text.lower()
                if 'bds' in text:
                    beds = float(text.split('bds')[0].strip())
                elif 'ba' in text:
                    baths = float(text.split('ba')[0].split()[-1])
                elif 'sqft' in text:
                    sqft = float(text.split('sqft')[0].replace(',', '').strip())
            
            return {
                'price': price,
                'beds': beds,
                'baths': baths,
                'sqft': sqft,
                'address': address
            }
            
        except Exception as e:
            print(f"Error parsing listing: {e}")
            return None
    
    def scrape(self, max_pages=5, delay=1):
        """Scrape multiple pages of listings."""
        for page in range(1, max_pages + 1):
            print(f"Scraping page {page}...")
            html = self.get_page(page)
            
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'html.parser')
            listings = soup.select('.list-card')  # Update selector as needed
            
            for listing in listings:
                listing_data = self.parse_listing(listing)
                if listing_data:
                    self.data.append(listing_data)
            
            # Be nice to the server
            time.sleep(delay + random.uniform(0, 1))
        
        return pd.DataFrame(self.data)

# Example usage
if __name__ == "__main__":
    # Note: This is a placeholder URL. You'll need to replace it with an actual real estate website
    # and update the selectors in the parse_listing method accordingly.
    base_url = "https://www.example-real-estate.com/for_sale"
    
    print("Starting web scraping...")
    scraper = RealEstateScraper(base_url)
    df = scraper.scrape(max_pages=2)  # Scrape 2 pages as an example
    
    # Save the data
    if not df.empty:
        output_file = "../datasets/scraped_housing_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nScraped {len(df)} listings. Data saved to {output_file}")
        
        # Basic data exploration
        print("\nData Summary:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df.isnull().sum())
    else:
        print("No data was scraped. Please check the website structure and selectors.")
        
    print("\nNote: This is a template. You'll need to update the selectors "
          "to match the actual website structure you're scraping.")
