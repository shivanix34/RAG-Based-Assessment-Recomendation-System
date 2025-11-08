import chromadb

PERSIST_DIR = "./chroma_store"
COLLECTION_NAME = "shl_assessments"

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)

count = collection.count()
print(f" Found {count} records in ChromaDB.")

sample = collection.peek(3)  # show a few samples
for i, meta in enumerate(sample["metadatas"], 1):
    print(f"\n🔹 Sample {i}: {meta.get('assessment_name', 'N/A')}")
    print(f"   URL: {meta.get('url', 'N/A')}")
    print(f"   Type: {meta.get('test_type', 'N/A')}")
