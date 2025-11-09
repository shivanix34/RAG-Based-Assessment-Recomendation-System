import os
import json
import time
import io
import csv
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from rag_core import get_recommendations
from job_rag_new import fetch_job_description, extract_skills_llm

app = FastAPI(title="SHL Assessment Recommendation API")

# --- CORS Middleware ---
# crucial for allowing your Streamlit frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://assessment-recomendation-frontend.onrender.com",
    "https://assessment-recomendation-backend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    query: str

class UrlRecommendRequest(BaseModel):
    url: str


def map_test_type(type_code: Union[str, List[str], None]) -> List[str]:
    if not type_code:
        return []
        
    if isinstance(type_code, list):
         return type_code
    
    mapping = {
        'K': "Knowledge & Skills",
        'P': "Personality & Behaviour",
        'A': "Ability & Aptitude",
        'B': "Biodata & Situational Judgement",
        'C': "Competencies",
        'D': "Development and 360",
        'E': "Assessment Exercises",
        'S': "Simulations"
    }
    
    code_str = str(type_code).upper()
    codes = [c.strip() for c in code_str.replace(',', ' ').split() if c.strip()]
    
    return [mapping.get(c, c) for c in codes]

def safe_duration(val: Any) -> Union[int, str]:
    if val is None or val == "":
        return "N/A"
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return "N/A"

def format_assessment_for_api(assessment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": assessment.get("url", ""),
        "name": assessment.get("assessment_name", ""),
        "adaptive_support": assessment.get("adaptive_support", "No"), # Default as it's missing in source
        "description": assessment.get("description", ""),
        "duration": safe_duration(assessment.get("length_minutes")),
        "remote_support": assessment.get("remote_testing", "Yes"),
        "test_type": map_test_type(assessment.get("test_type", ""))
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    if not request.query.strip():
         raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        core_response = get_recommendations(request.query)
        
        if core_response['status'] == 'error':
             print(f"RAG Core Error: {core_response.get('error_message')}")
             raise HTTPException(status_code=500, detail="Internal recommendation engine error.")

        formatted_recs = [
            format_assessment_for_api(rec) for rec in core_response.get('recommendations', [])
        ]

        return {"recommended_assessments": formatted_recs}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error in /recommend: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# --- ADDITIONAL ENDPOINTS (For your full Frontend features) ---

@app.post("/recommend/url")
async def recommend_from_url(request: UrlRecommendRequest):
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        # 1. Fetch JD
        job_data = fetch_job_description(url)
        jd_text = job_data.get("description", "")
        if not jd_text or len(jd_text) < 50:
             raise HTTPException(status_code=422, detail="Could not extract sufficient text from URL.")

        # 2. Extract skills
        extracted_skills = extract_skills_llm(job_data, top_k=20)
        skills_query = ", ".join(extracted_skills)

        # 3. Get recommendations
        core_response = get_recommendations(skills_query)
        
        if core_response['status'] == 'error':
             raise HTTPException(status_code=500, detail=core_response.get('error_message'))

        formatted_recs = [
             format_assessment_for_api(rec) for rec in core_response.get('recommendations', [])
        ]

        return {
            "source_url": url,
            "extracted_job_title": job_data.get('title'),
            "extracted_query": skills_query,
            "recommended_assessments": formatted_recs
        }

    except Exception as e:
        print(f"Error in URL processing: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process URL: {str(e)}")

@app.post("/recommend/file")
async def process_batch_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = file.filename
        
        # 1. Read file into DataFrame
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(contents))
        elif filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload .csv or .xlsx")

        # 2. Validate columns
        cols_lower = [str(c).lower().strip() for c in df.columns]
        if 'query' not in cols_lower:
             raise HTTPException(status_code=400, detail="Input file must have a 'Query' column.")
        
        # Find exact column name for 'Query'
        query_col_idx = cols_lower.index('query')
        query_col = df.columns[query_col_idx]

        # 3. Process rows
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Query', 'Assessment_url'])

        unique_queries = df[query_col].dropna().unique().tolist()
        
        for query in unique_queries:
            query_str = str(query).strip()
            if not query_str:
                continue
                
            response = get_recommendations(query_str)
            
            if response['status'] == 'success':
                recs = response.get('recommendations', [])
                if not recs:
                     writer.writerow([query_str, "No recommendations found"])
                else:
                    for rec in recs:
                        writer.writerow([query_str, rec.get('url', 'N/A')])
            else:
                 writer.writerow([query_str, "Error processing query"])
            
            time.sleep(0.05)

        # 4. Return as downloadable CSV
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=processed_results.csv"}
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting SHL Assessment API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)