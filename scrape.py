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

# Product links
links = {
    "Samsung Galaxy S24 Ultra Cell Phone, 512GB AI Smartphone": "https://amzn.in/d/fIZBs8E",
    "Moto G Play | 2024 | Unlocked | Made for US 4/64GB": "https://amzn.in/d/fIxKM4C",
    "Tracfone | Motorola Moto g Play 2024 | Locked ": "https://amzn.in/d/20YT7Lg",
    "OnePlus Nord N30 5G | Unlocked Dual-SIM Android Smart Phone": "https://amzn.in/d/c8eAWbF",
}

def scrape_product_data(link, driver, wait):
    product_data = {"reviews": []}

    try:
        driver.get(link)
        time.sleep(3)

        # Scrape product prices
        try:
            price_elem = wait.until(EC.presence_of_element_located(
                (By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]')
            ))
            product_data["selling_price"] = int("".join(price_elem.text.strip().split(",")))
        except:
            product_data["selling_price"] = 0

        try:
            original_price_elem = driver.find_element(
                By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]'
            )
            product_data["original_price"] = int("".join(original_price_elem.text.strip().split(",")))
        except:
            product_data["original_price"] = 0

        # Scrape reviews with pagination
        try:
            reviews_link = driver.find_element(By.ID, "reviews-medley-footer").find_element(By.TAG_NAME, "a").get_attribute("href")
            driver.get(reviews_link)
            time.sleep(3)
            while True:
                reviews = driver.find_elements(By.CSS_SELECTOR, ".review-text-content span")
                for review in reviews:
                    product_data["reviews"].append(review.text.strip())

                # Navigate to the next page of reviews
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, ".a-pagination .a-last a")
                    if "a-disabled" in next_button.get_attribute("class"):
                        break
                    next_button.click()
                    time.sleep(2)
                except:
                    break
        except Exception as e:
            print(f"Error scraping reviews: {e}")

    except Exception as e:
        print(f"Error scraping link {link}: {e}")

    return product_data


# Main execution
if __name__ == "__main__":
    # Set up the Chrome WebDriver
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    wait = WebDriverWait(driver, 10)

    all_reviews = []
    all_prices = []

    for product_name, link in links.items():
        if not link:
            print(f"Skipping {product_name} due to empty link.")
            continue

        print(f"Scraping data for {product_name}...")
        product_data = scrape_product_data(link, driver, wait)
        all_prices.append({
            "product_name": product_name,
            "Price": product_data.get("selling_price", 0),
            "Discount": product_data.get("original_price", 0),
            "Date": datetime.now().strftime("%Y-%m-%d"),
        })
        for review in product_data.get("reviews", []):
            all_reviews.append({"product_name": product_name, "review": review})

    # Save data to CSV files
   
    pd.DataFrame(all_reviews).to_csv(f"reviews.csv", index=False)
    pd.DataFrame(all_prices).to_csv(f"competitor_data.csv", index=False)

    print("Scraping complete. Data saved to CSV files.")
    driver.quit()



