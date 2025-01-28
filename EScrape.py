import json
import time
from datetime import datetime
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# Dictionary containing product names and their corresponding Amazon links

links = {
    "Motorola razr | 2023 | Unlocked | Made for US 8/128 | 32MP Camera | Sage Green, 73.95 x 170.82 x 7.35mm amazon": "https://www.amazon.com/Motorola-Unlocked-Camera-170-82-7-35mm/dp/B0CGVXZSQJ?th=1",
    "Moto G Power 5G | 2024 | Unlocked | Made for US 8+128GB | 50MP Camera | Pale Lilac": "https://www.amazon.com/Power-Unlocked-128GB-Camera-Lilac/dp/B0CVR23QCR?th=1",
    "Tracfone | Motorola Moto g Play 2024 | Locked | 64GB | 5000mAh Battery | 50MP Quad Pixel Camera | 6.5-in. HD+ 90Hz Display | Sapphire Blue": "https://www.amazon.com/Tracfone-Motorola-5000mAh-Battery-Sapphire/dp/B0CTW8TXGH",
    
}

def scrape_product_data(link):  # Function to scrape product data (price, reviews, etc.) from an Amazon product page
    # Set up Chrome options for headless browsing
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en")
    options.add_argument("--window-size=1920,1080")

    # Initialize the Chrome WebDriver with options and automatic driver installation
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.set_window_size(1920, 1080)  # Set window size to full HD
    driver.get(link)   # Navigate to the provided product link

    
    product_data, review_data = {}, {}
    product_data["reviews"] = []       # Initialize an empty list to store reviews
    
    wait = WebDriverWait(driver, 10)
    time.sleep(5)   # Wait for 5 seconds before proceeding
    retry = 0
    
    while retry < 3:  # Retry logic to load the page if the product details don't load immediately
        try:
            driver.save_screenshot("screenshot.png")
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))
            break
        except Exception:
            print("retrying")
            retry += 1
            driver.get(link)  # Reload the page and wait for the element again
            time.sleep(5)

    driver.save_screenshot("screenshot.png")

    # Try to extract product price (selling price) from the page
    try:
        price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]',
        )
        # Clean and convert the price text to an integer
        product_data["selling_price"] = int("".join(price_elem.text.strip().split(",")))
        
    except:
        product_data["selling_price"] = 0 # Set to 0 if price is not found

    try:
        original_price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]',
        )
        product_data["original_price"] = int("".join(original_price_elem.text.strip().split(",")))
    except:
        product_data["original_price"] = 0
    
    try:
        discount = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]',
        )
        full_rating_text = discount.get_attribute("innerHTML").strip()
        if "out of 5 stars" in full_rating_text.lower():
            product_data["rating"] = (
                full_rating_text.lower().split(" out of")[0].strip()
            )
        else:
            product_data["discount"] = full_rating_text
    except:
        product_data["discount"] = 0

    try:
        driver.find_element(By.CLASS_NAME, "a-icon-popover").click()
        time.sleep(1)
    except:
        pass
        
# Try to scrape the reviews popover to open reviews section
    try:    
        reviews_link = driver.find_elements(
            By.XPATH, "//a[contains(text(), 'See customer reviews')]"
        )[1].get_attribute("href")   # review url
        product_data["product_url"] = reviews_link.split("#")[0]
        driver.get(reviews_link) # Navigate to reviews page
        time.sleep(3)
        reviews = driver.find_element(By.ID, "cm-cr-dp-review-list")  # Find reviews sections in the list
        reviews = reviews.find_elements(By.TAG_NAME, "li")  # extract reviews from the list
        for item in reviews:
            product_data["reviews"].append(item.get_attribute("innerText")) # Save each review's text
        driver.back()  # Go back to the original product page
    except Exception:
        product_data["reviews"] = []    # empty review list as an exception

    product_data["date"] = time.strftime("%Y-%m-%d") # Record the date of data collection
    review_data["date"] = time.strftime("%Y-%m-%d")
    driver.quit()
    return product_data # Return the scraped product data


for product_name, link in links.items():
        product_data = scrape_product_data(link)
    # Load existing reviews and price data from CSV files
        reviews = json.loads(pd.read_csv("reviews.csv").to_json(orient="records"))
        price = json.loads(pd.read_csv("competitor_data.csv").to_json(orient="records"))
        price.append({
            "product_name": product_name,
            "Price": product_data["selling_price"],
            "Discount": product_data["discount"],
            "Date": datetime.now().strftime("%Y-%m-%d"),
        })# Appended the new product price data

        for i in product_data["reviews"]:
            reviews.append({"product_name": product_name, "reviews": i})

        pd.DataFrame(reviews).to_csv("reviews.csv", index=False)
        pd.DataFrame(price).to_csv("competitor_data.csv", index=False)
print("Scraping complete. Data saved to CSV files.") # Save the updated reviews and price data back to CSV files


