#!/usr/bin/env python3
"""
Downloads the TMDB 5000 Movie Dataset from a public mirror.
Run once before launching the app.
"""
import os, urllib.request, zipfile, io

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "tmdb_5000_movies.csv": (
        "https://raw.githubusercontent.com/erkansirin78/datasets/master/tmdb_5000_movies.csv"
    ),
    "tmdb_5000_credits.csv": (
        "https://raw.githubusercontent.com/erkansirin78/datasets/master/tmdb_5000_credits.csv"
    ),
}

for fname, url in FILES.items():
    dest = os.path.join(DATA_DIR, fname)
    if os.path.exists(dest):
        print(f"  ✓ {fname} already exists, skipping.")
        continue
    print(f"  ↓ Downloading {fname}…")
    urllib.request.urlretrieve(url, dest)
    print(f"  ✓ Saved to {dest}")

print("\nAll data files ready. Run:  streamlit run app.py")
