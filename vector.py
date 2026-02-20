# ==========================
# vector.py - Build Vector Store
# ==========================
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import json
import os

CSV_PATH   = "MoviesOnStreamingPlatforms.csv"   # FIX: must match main.py
INDEX_PATH = "vector.index"
META_PATH  = "metadata.pkl"
STATS_PATH = "dataset_stats.json"

def build_vector_store():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Save dataset-level stats for the UI dashboard
    stats = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "numeric_summary": {}
    }
    for col in df.select_dtypes(include=[np.number]).columns:
        stats["numeric_summary"][col] = {
            "min":    float(df[col].min()),
            "max":    float(df[col].max()),
            "mean":   round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
        }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset stats saved: {len(df)} rows, {len(df.columns)} columns")

    # Build rich text chunks: one per row, with column names
    def row_to_text(row):
        parts = [f"{col}: {val}" for col, val in row.items()]
        return " | ".join(parts)

    texts   = df.apply(row_to_text, axis=1).tolist()
    records = df.to_dict(orient="records")   # store original rows for structured display

    print("Encoding embeddings...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    # IndexFlatIP + L2-normalised vectors = cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"texts": texts, "records": records}, f)

    print(f"✅ Vector DB created: {len(texts)} vectors, dim={dimension}")

if __name__ == "__main__":
    build_vector_store()
