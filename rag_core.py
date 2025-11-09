import os
import json
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import time

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

# ChromaDB setup
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "shl_assessments"

def get_gemini_model():
    """Initialize and return the Gemini model"""
    return genai.GenerativeModel('gemini-2.5-flash')

def analyze_query_focus(query: str) -> Dict:
    """Analyze query to determine technical vs behavioral focus"""
    model = get_gemini_model()
    
    prompt = f"""
Analyze this hiring query and determine the PRIMARY focus:

Query: {query}

Determine:
1. Is this query PRIMARILY about:
   - TECHNICAL skills (programming, tools, specific technical knowledge)
   - BEHAVIORAL/SOFT skills (personality, culture fit, leadership, communication)
   - BALANCED (equal emphasis on both)

2. What's the specificity level:
   - HIGHLY_SPECIFIC (mentions exact technologies, tools, or very specific behavioral traits)
   - MODERATE (general role requirements)
   - BROAD (general role title only)

3. Extract key requirements

Format EXACTLY as:
PRIMARY_FOCUS: TECHNICAL or BEHAVIORAL or BALANCED
SPECIFICITY: HIGHLY_SPECIFIC or MODERATE or BROAD
TECHNICAL_SKILLS: list of technical skills (or "None" if behavioral focus)
SOFT_SKILLS: list of soft skills (or "None" if technical focus)
JOB_LEVEL: Entry/Junior/Mid-level/Senior/Executive/Manager or "Not specified"
DURATION_MAX: number or "Not specified"
"""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        time.sleep(1)  # Rate limit protection
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {
            'primary_focus': 'BALANCED',
            'specificity': 'MODERATE',
            'technical_skills': query,
            'soft_skills': 'communication, teamwork',
            'job_level': None,
            'duration_max': None
        }

    # Parse response
    analysis = {
        'primary_focus': 'BALANCED',
        'specificity': 'MODERATE',
        'technical_skills': '',
        'soft_skills': '',
        'job_level': None,
        'duration_max': None
    }
    
    for line in response_text.split('\n'):
        line = line.strip()
        if line.startswith('PRIMARY_FOCUS:'):
            focus = line.replace('PRIMARY_FOCUS:', '').strip()
            if focus in ['TECHNICAL', 'BEHAVIORAL', 'BALANCED']:
                analysis['primary_focus'] = focus
        elif line.startswith('SPECIFICITY:'):
            spec = line.replace('SPECIFICITY:', '').strip()
            if spec in ['HIGHLY_SPECIFIC', 'MODERATE', 'BROAD']:
                analysis['specificity'] = spec
        elif line.startswith('TECHNICAL_SKILLS:'):
            analysis['technical_skills'] = line.replace('TECHNICAL_SKILLS:', '').strip()
        elif line.startswith('SOFT_SKILLS:'):
            analysis['soft_skills'] = line.replace('SOFT_SKILLS:', '').strip()
        elif line.startswith('JOB_LEVEL:'):
            level = line.replace('JOB_LEVEL:', '').strip()
            if level and level.lower() != "not specified":
                analysis['job_level'] = level
        elif line.startswith('DURATION_MAX:'):
            duration = line.replace('DURATION_MAX:', '').strip()
            if duration and duration.lower() != "not specified":
                try:
                    analysis['duration_max'] = int(duration.split()[0])
                except:
                    pass
    
    # Fallbacks
    if not analysis['technical_skills'] or analysis['technical_skills'].lower() == 'none':
        analysis['technical_skills'] = query
    if not analysis['soft_skills'] or analysis['soft_skills'].lower() == 'none':
        analysis['soft_skills'] = 'communication, collaboration, teamwork'
    
    print(f"Query Analysis: Focus={analysis['primary_focus']}, Specificity={analysis['specificity']}")
    print(f"Technical: {analysis['technical_skills'][:60]}...")
    print(f"Soft Skills: {analysis['soft_skills'][:60]}...")
    
    return analysis

def apply_metadata_filters(assessments: List[Dict], duration_max: int) -> List[Dict]:
    """Apply metadata filters - duration is HARD constraint"""
    if not duration_max:
        return assessments
    
    filtered = []
    for assessment in assessments:
        try:
            assessment_duration = assessment.get('length_minutes', '')
            if assessment_duration and str(assessment_duration).replace('.', '').isdigit():
                duration = float(assessment_duration)
                if duration <= duration_max:
                    filtered.append(assessment)
            else:
                # Include if duration not specified
                filtered.append(assessment)
        except:
            filtered.append(assessment)
    
    return filtered

def adaptive_threshold_selection(assessments: List[Dict], 
                                 assessment_type: str,
                                 min_count: int,
                                 max_count: int,
                                 start_threshold: float = 0.90) -> List[Dict]:
    """
    Iteratively lower threshold from 0.90 until we get min_count assessments
    """
    if not assessments:
        return []
    
    threshold = start_threshold
    min_threshold = 0.45
    step = 0.05
    
    while threshold >= min_threshold:
        filtered = [a for a in assessments if a['similarity_score'] >= threshold]
        
        if len(filtered) >= min_count:
            print(f"  {assessment_type}: threshold={threshold:.2f}, found {len(filtered)}")
            return filtered[:max_count]
        
        threshold -= step
    
    # If still not enough, return top min_count
    print(f"  {assessment_type}: Using top {min(min_count, len(assessments))} (no threshold met)")
    return assessments[:min_count]

def search_assessments(query_analysis: Dict,
                       min_total: int = 5, 
                       max_total: int = 10) -> Tuple[List[Dict], Dict]:
    """
    Smart search with strict thresholds and guaranteed minimums
    """
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5"
        )
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn
        )
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        raise
    
    primary_focus = query_analysis['primary_focus']
    specificity = query_analysis['specificity']
    technical_query = query_analysis['technical_skills']
    soft_skills_query = query_analysis['soft_skills']
    duration_max = query_analysis['duration_max']
    
    # Search for K and P assessments
    k_results = collection.query(
        query_texts=[technical_query],
        n_results=50,
        where={"test_type": "K"}
    )
    
    p_results = collection.query(
        query_texts=[soft_skills_query],
        n_results=50,
        where={"test_type": "P"}
    )
    
    # Process K-type results
    k_assessments = []
    if k_results['documents'] and k_results['documents'][0]:
        for idx, (doc, metadata, distance) in enumerate(zip(
            k_results['documents'][0],
            k_results['metadatas'][0],
            k_results['distances'][0]
        )):
            similarity = 1 - (distance / 2)
            assessment = {
                'rank': idx + 1,
                'similarity_score': round(similarity, 4),
                'assessment_name': metadata.get('assessment_name', ''),
                'test_type': metadata.get('test_type', ''),
                'description': metadata.get('description', ''),
                'url': metadata.get('url', ''),
                'job_levels': metadata.get('job_levels', ''),
                'length_minutes': metadata.get('assessment_length_(mins)', ''),
                'remote_testing': metadata.get('remote_testing', '')
            }
            k_assessments.append(assessment)
    
    # Process P-type results
    p_assessments = []
    if p_results['documents'] and p_results['documents'][0]:
        for idx, (doc, metadata, distance) in enumerate(zip(
            p_results['documents'][0],
            p_results['metadatas'][0],
            p_results['distances'][0]
        )):
            similarity = 1 - (distance / 2)
            assessment = {
                'rank': idx + 1,
                'similarity_score': round(similarity, 4),
                'assessment_name': metadata.get('assessment_name', ''),
                'test_type': metadata.get('test_type', ''),
                'description': metadata.get('description', ''),
                'url': metadata.get('url', ''),
                'job_levels': metadata.get('job_levels', ''),
                'length_minutes': metadata.get('assessment_length_(mins)', ''),
                'remote_testing': metadata.get('remote_testing', '')
            }
            p_assessments.append(assessment)
    
    # Apply duration filter (HARD constraint)
    k_assessments = apply_metadata_filters(k_assessments, duration_max)
    p_assessments = apply_metadata_filters(p_assessments, duration_max)
    
    # Sort by similarity
    k_assessments.sort(key=lambda x: x['similarity_score'], reverse=True)
    p_assessments.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"\nAvailable after filters - K: {len(k_assessments)}, P: {len(p_assessments)}")
    
    # DETERMINE TARGET DISTRIBUTION based on query focus
    if primary_focus == 'TECHNICAL':
        if specificity == 'HIGHLY_SPECIFIC':
            min_k, max_k = 6, 9
            min_p, max_p = 1, 2
        else:
            min_k, max_k = 4, 7
            min_p, max_p = 1, 3
    elif primary_focus == 'BEHAVIORAL':
        if specificity == 'HIGHLY_SPECIFIC':
            min_k, max_k = 1, 2
            min_p, max_p = 6, 9
        else:
            min_k, max_k = 1, 3
            min_p, max_p = 4, 7
    else:  # BALANCED
        min_k, max_k = 3, 6
        min_p, max_p = 3, 6
    
    print(f"Target distribution - K: {min_k}-{max_k}, P: {min_p}-{max_p}")
    
    # ADAPTIVE THRESHOLD SELECTION
    selected_k = adaptive_threshold_selection(k_assessments, 'K', min_k, max_k, start_threshold=0.90)
    selected_p = adaptive_threshold_selection(p_assessments, 'P', min_p, max_p, start_threshold=0.90)
    
    # GUARANTEE MINIMUM REQUIREMENTS: At least 1 K and 1 P
    if len(selected_k) == 0 and len(k_assessments) > 0:
        selected_k = [k_assessments[0]]
        print("  Forcing minimum 1 K assessment")
    
    if len(selected_p) == 0 and len(p_assessments) > 0:
        selected_p = [p_assessments[0]]
        print("  Forcing minimum 1 P assessment")
    
    # Combine and sort by similarity
    combined = selected_k + selected_p
    combined.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Remove duplicates
    seen = set()
    unique_assessments = []
    for assessment in combined:
        name = assessment['assessment_name']
        if name not in seen:
            seen.add(name)
            unique_assessments.append(assessment)
    
    # GUARANTEE MINIMUM TOTAL: At least min_total assessments
    if len(unique_assessments) < min_total:
        print(f"  Need more assessments ({len(unique_assessments)} < {min_total})")
        remaining = []
        seen_names = {a['assessment_name'] for a in unique_assessments}
        
        # Pool remaining assessments prioritized by focus
        if primary_focus == 'TECHNICAL':
            pool = k_assessments + p_assessments
        elif primary_focus == 'BEHAVIORAL':
            pool = p_assessments + k_assessments
        else:
            pool = k_assessments + p_assessments
        
        for a in pool:
            if a['assessment_name'] not in seen_names:
                remaining.append(a)
        
        remaining.sort(key=lambda x: x['similarity_score'], reverse=True)
        needed = min_total - len(unique_assessments)
        unique_assessments.extend(remaining[:needed])
    
    # Apply max cap
    unique_assessments = unique_assessments[:max_total]
    
    # Final verification: ensure at least 1 K and 1 P
    final_k = sum(1 for a in unique_assessments if a['test_type'].upper() == 'K')
    final_p = sum(1 for a in unique_assessments if a['test_type'].upper() == 'P')
    
    # Emergency fix: if somehow we still don't have both types
    if final_k == 0 and len(k_assessments) > 0:
        # Remove lowest P, add top K
        unique_assessments = [a for a in unique_assessments if a['test_type'].upper() != 'P'][:max_total-1]
        unique_assessments.append(k_assessments[0])
        print("  Emergency: Added K assessment")
    
    if final_p == 0 and len(p_assessments) > 0:
        # Remove lowest K, add top P
        unique_assessments = [a for a in unique_assessments if a['test_type'].upper() != 'K'][:max_total-1]
        unique_assessments.append(p_assessments[0])
        print("  Emergency: Added P assessment")
    
    # Re-sort and recalculate
    unique_assessments.sort(key=lambda x: x['similarity_score'], reverse=True)
    final_k = sum(1 for a in unique_assessments if a['test_type'].upper() == 'K')
    final_p = sum(1 for a in unique_assessments if a['test_type'].upper() == 'P')
    
    print(f"Final Selection - Total: {len(unique_assessments)}, K: {final_k}, P: {final_p}")
    
    distribution_info = {
        'total_results': len(unique_assessments),
        'knowledge_test_count': final_k,
        'personality_test_count': final_p,
        'primary_focus': primary_focus
    }
    
    return unique_assessments, distribution_info

def get_recommendations(query: str) -> Dict:
    """Main function to get assessment recommendations"""
    try:
        query_analysis = analyze_query_focus(query)
        
        assessments, distribution = search_assessments(
            query_analysis,
            min_total=5,
            max_total=10
        )
        
        response = {
            'original_query': query,
            'query_analysis': query_analysis,
            'recommendations': assessments,
            'distribution': distribution,
            'status': 'success'
        }
        
        return response
        
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'original_query': query
        }

if __name__ == "__main__":
    # Example usage
    sample_query = "Looking for a Senior Java Developer with Spring Boot experience and strong team leadership skills. Must complete assessment within 40 minutes."
    result = get_recommendations(sample_query)
    print(json.dumps(result, indent=2))