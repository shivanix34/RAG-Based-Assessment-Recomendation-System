import streamlit as st
import requests
import pandas as pd
import io
import json
import os

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# --- THEME COLORS ---
SHL_PRIMARY_BLUE = "#0077B5"
SHL_HOVER_BLUE = "#005E93"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
BORDER_COLOR = "#CCCCCC"
LIGHT_BLUE_BG = "#F0F7FF" # Fixed: truly light blue for backgrounds/hover

# --- Page Config & Styling ---
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown(f"""
    <style>
    /* Main Background */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .main .block-container {{
        background-color: {BACKGROUND_COLOR};
        padding-top: 2rem;
        max-width: 1000px;
    }}
    
    /* Headers & Text */
    h1, h2, h3, h4, h5, h6, p, div, label, span {{
        color: {TEXT_COLOR} !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}
    h1 {{
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    /* Standard Buttons (Blue) */
    .stButton > button {{
        background-color: {SHL_PRIMARY_BLUE} !important;
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }}
    .stButton > button:hover {{
        background-color: {SHL_HOVER_BLUE} !important;
        color: white !important;
    }}

    /* SPECIFIC STYLE FOR DOWNLOAD BUTTON (Light Green) */
    [data-testid='stDownloadButton'] > button {{
        background-color: #22C55E !important; /* Light Green */
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border-radius: 4px;
    }}
    [data-testid='stDownloadButton'] > button:hover {{
        background-color: #16A34A !important; /* Darker Green Hover */
        color: white !important;
    }}

    /* --- CUSTOM FILE UPLOADER STYLING (Light Green Theme) --- */
    [data-testid='stFileUploaderDropzone'] {{
        background-color: #F0FDF4 !important; /* Very light green background */
        border: 2px dashed #4ADE80 !important; /* Light green dashed border */
        border-radius: 8px;
    }}
    [data-testid='stFileUploaderDropzone'] div {{
        color: #15803D !important; /* Darker green text inside dropzone */
    }}
    [data-testid='stFileUploaderDropzone'] button {{
        background-color: #22C55E !important; /* Green 'Browse files' button */
        color: white !important;
        border: none !important;
    }}
    [data-testid='stFileUploaderDropzone'] button:hover {{
        background-color: #16A34A !important; /* Darker green on hover */
    }}
    /* ------------------------------------------------------- */

    /* Input Fields */
    .stTextArea textarea, .stTextInput input {{
        background-color: #FFFFFF !important;
        color: {TEXT_COLOR} !important;
        caret-color: {TEXT_COLOR} !important; /* Force cursor to be visible */
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 4px;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {SHL_PRIMARY_BLUE} !important;
        box-shadow: 0 0 0 1px {SHL_PRIMARY_BLUE} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        border-bottom: 1px solid #F0F0F0; /* Restored gray baseline */
    }}
    .stTabs [data-baseweb="tab"] {{
        height: auto;
        padding-bottom: 12px;
        background-color: transparent;
        border: none;
        color: #666666 !important;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: transparent !important;
        color: {SHL_PRIMARY_BLUE} !important;
    }}
    /* Hides the default red decoration line Streamlit sometimes adds */
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {SHL_PRIMARY_BLUE} !important;
    }}

    /* Assessment Cards */
    .assessment-card {{
        background-color: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    .assessment-card:hover {{
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-color: {SHL_PRIMARY_BLUE};
        transform: translateY(-2px);
    }}
    .assessment-title a {{
        color: {SHL_PRIMARY_BLUE} !important;
        text-decoration: none;
        font-size: 1.3rem;
        font-weight: 700;
    }}
    .assessment-title a:hover {{
        text-decoration: underline;
    }}
    .meta-info {{
        font-size: 0.95rem;
        color: #555555 !important;
        margin: 12px 0;
    }}
    .tag {{
        display: inline-block;
        background-color: {LIGHT_BLUE_BG};
        color: {SHL_PRIMARY_BLUE} !important;
        padding: 5px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 8px;
        border: 1px solid #D6E9FF;
    }}
    
    /* Alerts/Info Boxes */
    .stAlert {{
        background-color: #F8F9FA;
        color: {TEXT_COLOR};
        border: 1px solid {BORDER_COLOR};
    }}

    /* EXPANDER STYLING */
    .streamlit-expanderHeader {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
    }}
    .streamlit-expanderHeader:hover {{
        background-color: {LIGHT_BLUE_BG} !important; /* Light blue on hover */
        color: {SHL_PRIMARY_BLUE} !important;
        border-color: {SHL_PRIMARY_BLUE} !important;
    }}
    /* Fix for dark state when focused but NOT hovered - revert to white */
    .streamlit-expanderHeader:focus:not(:hover),
    details[open] .streamlit-expanderHeader:not(:hover) {{
         background-color: {BACKGROUND_COLOR} !important;
         color: {SHL_HOVER_BLUE} !important;
         box-shadow: none !important;
    }}
    /* Content inside expander */
    .streamlit-expanderContent {{
        border: 1px solid {BORDER_COLOR};
        border-top: none;
        border-bottom-left-radius: 4px;
        border-bottom-right-radius: 4px;
        padding: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1 style='text-align: center;'>SHL Assessment Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-top: -10px; margin-bottom: 30px;'>AI-powered search for the perfect candidate assessment</p>", unsafe_allow_html=True)

# --- Main App Logic ---
tab1, tab2, tab3 = st.tabs(["Search by Query", "Batch Upload", "Search by Job URL"])

# --- TAB 1: Text Query ---
with tab1:
    st.write("##### Search by natural language")
    text_query = st.text_area("Describe the role, skills, or behaviors you are hiring for...", height=120, placeholder="e.g., Senior Java Developer who is a strong team player and can handle fast-paced environments...")

    # Use columns to align the button to the left
    col1, _ = st.columns([1, 5])
    with col1:
        search_clicked = st.button("Search", key="text_search_btn", use_container_width=True)

    if search_clicked:
        if not text_query.strip():
            st.warning("⚠️ Please enter hiring criteria first.")
        else:
            with st.spinner("Searching catalog..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/recommend", json={"query": text_query})
                    if response.status_code == 200:
                        results = response.json().get("recommended_assessments", [])
                        if not results:
                            st.info("No relevant assessments found for this specific query.")
                        else:
                            st.success(f"Found {len(results)} relevant assessments")
                            for rec in results:
                                st.markdown(f"""
                                <div class="assessment-card">
                                    <div class="assessment-title">
                                        <a href="{rec.get('url', '#')}" target="_blank">{rec.get('name', 'Unnamed Assessment')}</a>
                                    </div>
                                    <p style="margin-top: 10px; color: #444;">{rec.get('description', 'No description available.')}</p>
                                    <div class="meta-info">
                                        ⏱️ <strong>Duration:</strong> {rec.get('duration', 'N/A')} mins &nbsp;|&nbsp; 
                                        🌍 <strong>Remote Ready:</strong> {rec.get('remote_support', 'N/A')}
                                    </div>
                                    <div>
                                        {' '.join([f'<span class="tag">{t}</span>' for t in rec.get('test_type', [])])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error(f"Search failed. Server responded with: {response.status_code}")
                except requests.exceptions.ConnectionError:
                     st.error("❌ Could not connect to backend. Is 'main.py' running?")

# --- TAB 2: File Upload ---
with tab2:
    st.write("##### Batch process multiple queries")
    st.info("Upload an Excel (.xlsx) or CSV file with a column named 'Query'.")
    uploaded_file = st.file_uploader(
        "Upload File",
        type=["csv", "xlsx"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("Process Batch", key="file_btn"):
            with st.spinner("Processing queries... This might take some time."):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/recommend/file", files=files)
                    
                    if response.status_code == 200:
                        st.success("✅ Processing complete!")
                        st.download_button(
                            label="Download Results CSV",
                            data=response.content,
                            file_name=f"SHL_Results_{uploaded_file.name}.csv",
                            mime="text/csv",
                            key="download_btn"
                        )
                    else:
                         st.error(f"Processing failed: {response.text}")
                except Exception as e:
                     st.error(f"Connection error: {e}")

# --- TAB 3: URL Input ---
with tab3:
    st.write("##### Search by Job Posting URL")
    url_query = st.text_input("Paste a LinkedIn or job board URL...", placeholder="https://www.linkedin.com/jobs/view/...")

    col1, _ = st.columns([1, 5])
    with col1:
        url_search_clicked = st.button("Search", key="url_btn", use_container_width=True)

    if url_search_clicked:
        if not url_query.strip():
             st.warning("⚠️ Please enter a valid URL.")
        else:
            with st.spinner("Analyzing job post..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/recommend/url", json={"url": url_query})
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader(f"Job: {data.get('extracted_job_title', 'Analyzed Role')}")
                        
                        with st.expander("See extracted skills"):
                            st.write(data.get("extracted_query"))
                        
                        results = data.get("recommended_assessments", [])
                        if results:
                            for rec in results:
                                st.markdown(f"""
                                <div class="assessment-card">
                                    <div class="assessment-title">
                                        <a href="{rec.get('url', '#')}" target="_blank">{rec.get('name', 'Unnamed Assessment')}</a>
                                    </div>
                                    <p style="margin-top: 10px; color: #444;">{rec.get('description', 'No description available.')}</p>
                                    <div class="meta-info">
                                        ⏱️ <strong>Duration:</strong> {rec.get('duration', 'N/A')} mins &nbsp;|&nbsp; 
                                        🌍 <strong>Remote Ready:</strong> {rec.get('remote_support', 'N/A')}
                                    </div>
                                     <div>
                                        {' '.join([f'<span class="tag">{t}</span>' for t in rec.get('test_type', [])])}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                             st.warning("Could not find relevant assessments for this job post.")
                    else:
                        st.error(f"Analysis failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #999; font-size: 0.8em;'>SHL Internal Tool Prototype</div>", unsafe_allow_html=True)