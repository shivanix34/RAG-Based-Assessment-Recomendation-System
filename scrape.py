from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from collections import defaultdict
import csv

chrome_options = Options()
# chrome_options.add_argument("--headless=new")  # optional for background run
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 40)
base_url = "https://www.shl.com"

count = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360, 372, 384]

table_data = defaultdict(list)
curr_table_name = ""

for i in count:
    url = f"https://www.shl.com/solutions/products/product-catalog/?start={i}&type=1"
    print(f"\n🔹 Scraping page: {url}")
    driver.get(url)

    # Accept cookies only once
    try:
        cookie_btn = wait.until(
            EC.element_to_be_clickable((By.ID, "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll"))
        )
        cookie_btn.click()
        print("✅ Cookie banner accepted.")
    except Exception:
        print("ℹ️ No cookie banner found.")

    # Wait for the product table to load
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".custom__table-responsive")))
    time.sleep(5)

    rows = driver.find_elements(By.CSS_SELECTOR, ".custom__table-responsive tbody tr")
    print(f"✅ Found {len(rows)} rows.\n")

    for row in rows:
        header_elem = row.find_elements(By.CSS_SELECTOR, "th.custom__table-heading__title")
        link_elem = row.find_elements(By.CSS_SELECTOR, "td.custom__table-heading__title a")

        # Identify header row
        if header_elem:
            curr_table_name = header_elem[0].text.strip()

        # Extract product rows (skip unwanted tables)
        elif link_elem and curr_table_name and curr_table_name != "Pre-packaged Job Solutions":
            name = link_elem[0].text.strip()
            href = link_elem[0].get_attribute("href")
            if href.startswith("/"):
                href = base_url + href

            # Get all columns in the row
            cols = row.find_elements(By.CSS_SELECTOR, "td.custom__table-heading__general")

            # Extract Remote Testing (2nd column)
            remote_testing = "Yes" if len(cols) > 0 and cols[0].find_elements(By.CSS_SELECTOR, "span.catalogue__circle.-yes") else "No"

            # Extract Adaptive/IRT (3rd column)
            adaptive_irt = "Yes" if len(cols) > 1 and cols[1].find_elements(By.CSS_SELECTOR, "span.catalogue__circle.-yes") else "No"

            table_data[curr_table_name].append((name, href, remote_testing, adaptive_irt))
            print(f"- {name} | Remote: {remote_testing} | Adaptive: {adaptive_irt}")

driver.quit()

# Save only "Individual Test Solutions" table
category_to_save = "Individual Test Solutions"
output_file = "Individual_Test_Solutions.csv"

if category_to_save in table_data:
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Assessment Name", "URL", "Remote Testing", "Adaptive/IRT"])
        for name, href, remote_testing, adaptive_irt in table_data[category_to_save]:
            writer.writerow([name, href, remote_testing, adaptive_irt])
    print(f"\n✅ Data saved to '{output_file}' ({len(table_data[category_to_save])} entries).")
else:
    print("\n⚠️ No 'Individual Test Solutions' data found!")