import os
import csv
import time
import json
import re
from urllib.parse import urlparse
from collections import Counter

import requests
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords

import google.generativeai as genai
from dotenv import load_dotenv

from typing import Dict, List
from rag_core import analyze_query_focus, search_assessments

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(" GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)
client = genai.GenerativeModel("gemini-2.5-flash")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from collections import Counter
import numpy as np


def extract_common_skills(skills: list[str], emb_model, top_k=15):
    """
    Dynamically determine 'common' or dominant skill clusters for the JD.
    Uses frequency + embedding similarity to group related skills.
    """
    # Step 1: Normalize and count frequency
    cleaned = [s.lower().strip() for s in skills]
    freq = Counter(cleaned)

    # Step 2: Compute embeddings for each skill
    embs = np.array([emb_model(s) for s in cleaned])

    # Step 3: Cluster by cosine similarity
    common_indices = []
    for i, e1 in enumerate(embs):
        sims = np.dot(embs, e1) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(e1) + 1e-6)
        if np.mean(sims) > 0.65:  # Adjust threshold
            common_indices.append(i)

    # Step 4: Select top-k frequent + central skills
    ranked = [skill for skill, _ in freq.most_common(top_k)]
    dynamic_common = list(set([cleaned[i] for i in common_indices] + ranked))
    return dynamic_common


def extract_job_data_llm(html: str) -> dict:
    """LLM-based fallback for extracting job data from raw HTML."""
    prompt = """
    You are an expert web parser.
    Given the HTML of a job page, extract:
    1. Job Title
    2. Company (if present)
    3. Job Description (main text)

    Return output in valid JSON with keys:
    title, company, description.

    Only return the JSON object, nothing else.
    """
    try:
        response = client.generate_content(f"{prompt}\n\nHTML:\n{html[:8000]}")
        text_output = response.text.strip().replace("```json", "").replace("```", "")
        job_data = json.loads(text_output)
        return job_data
    except Exception as e:
        print(f" LLM extraction failed: {e}.")
        return {"title": None, "company": None, "description": None}


def extract_skills_llm(job_data: dict, top_k: int = 25) -> list[str]:
    """Extract key technical + soft skills from structured job data using LLM."""
    title = job_data.get("title", "")
    description = job_data.get("description", "")

    prompt = f"""
    You are an expert HR and AI analyst.
    Analyze the following job posting and extract the top {top_k} most important skills.

    Focus on:
    - Technical skills (Python, ML, AI, NLP, TensorFlow, etc.)
    - Soft skills (communication, teamwork, problem-solving, leadership)
    - Job-specific domain expertise (e.g., robotics, research, automation)

    Job Title: {title}
    Job Description: {description[:7000]}

    Return strictly valid JSON:
    {{ "skills": ["skill1", "skill2", "skill3", ...] }}
    """
    try:
        response = client.generate_content(prompt)
        text_output = response.text.strip().replace("```json", "").replace("```", "")
        match = re.search(r"\{.*\}", text_output, re.DOTALL)
        if match:
            text_output = match.group(0)
        data = json.loads(text_output)
        return [s.strip() for s in data.get("skills", []) if len(s.strip()) > 2]
    except Exception as e:
        print(f" LLM skill extraction failed: {e}")
        return extract_common_skills(description, top_k)


def fetch_job_description(url: str, timeout: int = 10) -> dict:
    """
    Scrape a LinkedIn or similar job posting and extract:
    - Job title
    - Job description (from <div class="mt4"> or similar)
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch URL: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # --- Extract title ---
    title_tag = (
        soup.select_one("h1")
        or soup.select_one("h2[class*=top-card-layout__title]")
        or soup.find("title")
    )
    job_title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    # --- Extract JD from <div class="mt4"> or fallback containers ---
    jd_div = soup.select_one("div.mt4") or soup.select_one("div[class*=description]")
    if jd_div:
        jd_text = jd_div.get_text(separator="\n", strip=True)
        if len(jd_text.split()) > 50:
            print(" Extracted job description from <div class='mt4'> successfully.")
            return {"title": job_title, "description": jd_text}

    # Fallback to LLM extraction
    print(" Fallback to LLM-based extraction...")
    job_data = extract_job_data_llm(resp.text)
    if not job_data.get("title"):
        job_data["title"] = job_title
    return job_data


def get_recommendations_v2(skills_query: str, min_results: int = 5, max_results: int = 10) -> Dict:
    response = {
        'original_query': skills_query,
        'status': 'error',
        'recommendations': [],
        'query_analysis': {},
        'distribution': {},
        'error_message': None
    }

    try:
        query_analysis = analyze_query_focus(skills_query)
        response['query_analysis'] = query_analysis

        assessments, distribution = search_assessments(
            query_analysis,
            min_total=min_results,
            max_total=max_results
        )

        response.update({
            'recommendations': assessments,
            'distribution': distribution,
            'status': 'success'
        })

    except Exception as e:
        response['error_message'] = f"Unexpected error: {str(e)}"

    return response


def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace(":", "_")
    path = parsed.path.strip("/ ").replace("/", "_") or "root"
    timestamp = int(time.time())
    return f"{domain}_{path[:80]}_{timestamp}.csv"


def process_job_url(url: str, output_file: str | None = None, rate_limit: float = 1.0):
    print(f"\nFetching job posting: {url}")

    job_data = fetch_job_description(url)
    jd_text = job_data.get("description", "")

    if not jd_text or len(jd_text.strip()) < 30:
        raise RuntimeError("Extracted job description is too short or invalid.")

    print(f"Extracted Job Title: {job_data.get('title', 'N/A')}")
    print(f"Extracted {len(jd_text.split())} words from job description.\n")

    print("Extracting skills using LLM...")
    significant_skills = extract_skills_llm(job_data, top_k=25)
    print("Extracted skills/keywords:", ", ".join(significant_skills))

    print("\nRunning RAG pipeline using extracted skills only...")
    skills_query = ", ".join(significant_skills)
    print(f"Skills Query: {skills_query[:250]}{'...' if len(skills_query) > 250 else ''}")

    recommendations = get_recommendations_v2(skills_query)

    if output_file is None:
        filename = safe_filename_from_url(url)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        outdir = os.path.join(script_dir, "output")
        os.makedirs(outdir, exist_ok=True)
        output_file = os.path.join(outdir, filename)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Source_URL", "Assessment_url"])

        if recommendations.get("status") == "success" and recommendations.get("recommendations"):
            for rec in recommendations["recommendations"]:
                writer.writerow([url, rec.get("url", "N/A")])
        else:
            writer.writerow([url, f"Error: {recommendations.get('error_message', 'No results')}"])

    print(f"\nResults saved to: {output_file}\n")
    time.sleep(rate_limit)

    return recommendations


if __name__ == "__main__":
    TARGET_URL = "https://www.linkedin.com/jobs/view/4320030147/?alternateChannel=search&eBP=BUDGET_EXHAUSTED_JOB&refId=wlhEL1vcXGGn4QPg8ONk4g%3D%3D&trackingId=BGgDORGQuik9iDqaNzMqrQ%3D%3D"
    print("Running SHL RAG pipeline on single job posting...\n")
    try:
        process_job_url(TARGET_URL)
    except Exception as e:
        print("\nError during execution:")
        print(e)
