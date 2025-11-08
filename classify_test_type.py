import os
import re
import time
import random
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv


# -------------------- SETUP --------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(" Missing GOOGLE_API_KEY in .env file")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

INPUT_FILE = "SHL_Product_Details_Final.csv"
OUTPUT_FILE = "SHL_Product_Details_Final_Updated.csv"
CHUNK_SIZE = 30 


# -------------------- LOAD DATA --------------------
df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

if "test_type" not in df.columns:
    raise ValueError("CSV must contain a 'test_type' column")

print(f" Loaded {len(df)} rows from {INPUT_FILE}")


# -------------------- NORMALIZE EXISTING TAGS --------------------
def normalize_tags(val):
    if pd.isna(val):
        return ""
    val = str(val).upper().replace(" ", "")
    tags = list(dict.fromkeys([t for t in val.split(",") if t]))
    return ",".join(tags)


# -------------------- GEMINI CLASSIFIER --------------------
def classify_with_gemini(row, retries=3):
    prompt = f"""
    You are an expert evaluator trained in SHL-style assessment taxonomy.
    Your task is to classify each assessment into one of these categories:

    1. K (Knowledge/Skill-based) — measures technical, cognitive, or job-related skills.
    2. P (Personality/Behavior-based) — measures personality traits, leadership style, motivation, attitude, behavior, emotional intelligence, or culture fit.
    3. K,P (Both) — if it measures both technical/knowledge and behavioral/personality dimensions.

    --- INPUT DETAILS ---
    Assessment Name: {row.get('assessment_name', '')}
    Description: {row.get('description', '')}
    Job Levels: {row.get('job_levels', '')}
    Assessment Length (mins): {row.get('assessment_length_(mins)', '')}
    Remote Testing: {row.get('remote_testing', '')}
    Adaptive/IRT Support: {row.get('adaptive/irt_support', '')}
    Existing Test Type (raw): {row.get('test_type', '')}

    --- DECISION RULES ---
    - Skill/Technical/Analytical terms → K
    - Personality/Behavioral/Leadership terms → P
    - Both → K,P

    Respond ONLY with one label: K, P, or K,P
    """

    for attempt in range(retries):
        try:
            res = model.generate_content(prompt)
            label = res.text.strip().upper().replace(" ", "")
            if label not in ["K", "P", "K,P"]:
                label = "K"
            return label

        except Exception as e:
            err_msg = str(e)
            print(f" Gemini error (attempt {attempt+1}/{retries}): {err_msg}")

            # Detect rate-limit hint
            match = re.search(r"retry in (\d+\.?\d*)s", err_msg)
            if match:
                delay = float(match.group(1))
                print(f" API quota hit. Sleeping for {delay:.1f} seconds...")
                time.sleep(delay + 2)
            else:
                sleep_time = 2 ** attempt + random.uniform(0, 1)
                print(f" Backoff for {sleep_time:.1f}s...")
                time.sleep(sleep_time)

            if attempt == retries - 1:
                print(" Giving up after retries.")
                return "K"


# -------------------- MAIN CLASSIFICATION FUNCTION --------------------
def classify_and_update(row):
    current = normalize_tags(row["test_type"])
    tags = [t.strip().upper() for t in current.split(",") if t.strip()]
    has_k = "K" in tags
    has_p = "P" in tags

    # If already has K or P, keep existing
    if has_k or has_p:
        return ",".join(sorted(tags))

    # Otherwise, classify via Gemini
    new_label = classify_with_gemini(row)
    for tag in new_label.split(","):
        if tag not in tags:
            tags.append(tag)

    merged = ",".join(sorted(tags))
    print(f"Processed: {row.get('assessment_name','')} → {merged}")
    time.sleep(1)
    return merged


# -------------------- PROCESS IN CHUNKS --------------------
total_rows = len(df)
for i in range(0, total_rows, CHUNK_SIZE):
    chunk = df.iloc[i:i + CHUNK_SIZE].copy()
    print(f"\n🔹 Processing rows {i + 1} to {min(i + CHUNK_SIZE, total_rows)}...")

    # Update only test_type column
    chunk["test_type"] = chunk.apply(classify_and_update, axis=1)

    # Save incrementally (overwrite or append)
    mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
    header = not os.path.exists(OUTPUT_FILE)
    chunk.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)

    print(f" Saved progress to {OUTPUT_FILE}")
    print(" Sleeping 60s before next chunk...")
    time.sleep(60)

print(f"\n All done! Classified file saved as: {OUTPUT_FILE}")
