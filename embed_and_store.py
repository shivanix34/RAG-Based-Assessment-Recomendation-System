import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import shutil

CSV_PATH = "SHL_Product_Details_Final_Clean.csv"
PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "shl_assessments"

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f" Deleted old Chroma store at: {PERSIST_DIR}")

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Combine key fields into one searchable text
def combine_text(row):
    # Only include semantically meaningful content for embedding
    semantic_parts = [
        f"Assessment Name: {row.get('assessment_name', '')}",
        f"Description: {row.get('description', '')}",
        f"Job Levels: {row.get('job_levels', '')}",
        f"Test Type: {row.get('test_type', '')}"
    ]
    return " | ".join(str(p) for p in semantic_parts if p)

df["combined_text"] = df.apply(combine_text, axis=1)

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

documents = df["combined_text"].tolist()
ids = [str(i) for i in range(len(df))]
metadatas = df.to_dict(orient="records")  # includes URL, test type, etc.

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print(f" Stored {len(df)} assessments into persistent ChromaDB at: {PERSIST_DIR}")