import csv
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === Setup Chrome ===
chrome_options = Options()
# chrome_options.add_argument("--headless=new")  # Uncomment for silent run
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 20)

# === Input and Output files ===
input_file = "Individual_Test_Solutions.csv"
output_file = "SHL_Product_Details_Final.csv"

# === Read URLs and existing fields from input CSV ===
entries = []
with open(input_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        entries.append({
            "name": row["Assessment Name"].strip(),
            "url": row["URL"].strip(),
            "remote": row.get("Remote Testing", "").strip(),
            "adaptive": row.get("Adaptive/IRT", "").strip(),
        })

print(f"🔹 Loaded {len(entries)} entries to scrape.\n")

# === Prepare Output CSV ===
with open(output_file, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow([
        "Assessment Name",
        "URL",
        "Remote Testing",
        "Adaptive/IRT Support",
        "Description",
        "Job Levels",
        "Assessment Length (mins)",
        "Test Type"
    ])

    for idx, entry in enumerate(entries, start=1):
        url = entry["url"]
        print(f"[{idx}/{len(entries)}] Scraping: {url}")
        driver.get(url)

        # Wait for product info
        try:
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, ".product-catalogue-training-calendar__row")
            ))
        except Exception:
            print("⚠️ Page did not load properly, skipping.")
            writer.writerow([
                entry["name"], entry["url"], entry["remote"], entry["adaptive"],
                "", "", "", ""
            ])
            continue

        time.sleep(2)

        def get_text_by_label(label):
            """Find <h4> label and return text from next <p>"""
            try:
                h4 = driver.find_element(
                    By.XPATH,
                    f"//h4[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{label.lower()}')]"
                )
                p = h4.find_element(By.XPATH, "following-sibling::p")
                return p.text.strip()
            except Exception:
                return ""

        # Extract fields
        description = get_text_by_label("Description")
        job_levels = get_text_by_label("Job levels")

        # Extract numeric duration (e.g., "10" from "Approximate Completion Time in minutes = 10")
        duration_text = get_text_by_label("Assessment length")
        match = re.search(r"\b\d+\b", duration_text)
        duration = match.group(0) if match else ""

        # Extract all test type tags (K, D, P, etc.)
        test_type_elems = driver.find_elements(By.CSS_SELECTOR, "span.product-catalogue__key")
        test_type = ", ".join([elem.text.strip() for elem in test_type_elems if elem.text.strip()])

        print(f"  → {entry['name']} | Job: {job_levels or 'N/A'} | Type: {test_type or 'N/A'} | Duration: {duration or 'N/A'} mins")

        # Write to CSV
        writer.writerow([
            entry["name"],
            entry["url"],
            entry["remote"],
            entry["adaptive"],
            description,
            job_levels,
            duration,
            test_type
        ])

driver.quit()
print(f"\n✅ Done! All details saved in '{output_file}'")
