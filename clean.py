import pandas as pd

# Load the CSV
df = pd.read_csv("SHL_Product_Details_Final_Updated.csv")

print(" Original rows:", len(df))

# Check duplicates based on all columns (exact duplicates)
dupes = df[df.duplicated()]
print(" Duplicate rows found:", len(dupes))

# Optionally preview some
if len(dupes) > 0:
    print("\n Sample duplicates:")
    print(dupes.head(5))

# Remove duplicates
df = df.drop_duplicates()

print(" Cleaned rows:", len(df))

# Overwrite cleaned file
df.to_csv("SHL_Product_Details_Final_Clean.csv", index=False)
print(" Saved cleaned CSV → SHL_Product_Details_Final_Clean.csv")
