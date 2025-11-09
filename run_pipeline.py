import pandas as pd
import csv
import time
from rag_core import get_recommendations

def process_dataset(input_filename: str, output_filename: str):
    """Process a dataset of queries and generate assessment recommendations"""
    
    print(f"Starting assessment recommendation process...")
    print(f"Input file: {input_filename}")
    print(f"Output file: {output_filename}")
    
    queries = []
    
    try:
        try:
            df = pd.read_excel(input_filename, sheet_name='Test-Set')
        except:
            df = pd.read_excel(input_filename, sheet_name=0)
            
        if "Query" in df.columns:
            queries = df['Query'].dropna().unique().tolist()
            print(f"Found {len(queries)} unique queries to process.")
        else:
            print(f"Error: 'Query' column not found")
            queries = []
            
    except Exception as e:
        print(f"Error reading input file: {e}")
        queries = []

    if queries:
        try:
            with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Query', 'Assessment_url', 'Test_Type', 'Similarity_Score'])
                
                for i, query in enumerate(queries):
                    if not query.strip():
                        continue
                        
                    print("\n" + "="*80)
                    print(f"Processing query {i+1}/{len(queries)}: {query[:70]}...")
                    
                    recommendations = get_recommendations(query)
                    
                    if recommendations['status'] == 'success':
                        if recommendations['recommendations']:
                            dist = recommendations['distribution']
                            print(f"✓ Found {dist['total_results']} (K:{dist['knowledge_test_count']}, P:{dist['personality_test_count']})")
                            for rec in recommendations['recommendations']:
                                writer.writerow([
                                    query,
                                    rec.get('url', 'N/A')
                                ])
                        else:
                            print("⚠ No recommendations found")
                            writer.writerow([query, 'No recommendations found', 'N/A', 0.0])
                    else:
                        print(f"✗ ERROR: {recommendations['error_message']}")
                        writer.writerow([query, f"Error: {recommendations['error_message']}", 'N/A', 0.0])
                    
                    # Rate limiting
                    time.sleep(1)
            
            print("\n" + "="*80)
            print(f"✓ Processing complete. Output saved to {output_filename}")

        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
    else:
        print("No queries found to process.")

if __name__ == "__main__":
    # Configuration
    input_file = "data/Gen_AI Dataset.xlsx"
    output_file = "output/Final_Test_Output.csv"
    
    # Process the dataset
    process_dataset(input_file, output_file)