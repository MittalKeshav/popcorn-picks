# 🎬 Popcorn Picks — Movie Recommendation System

A content-based movie recommender built with Streamlit, powered by the **TMDB 5000 Movie Dataset** (~5,000 films).

## Features
- **Content-based recommendations** (TF-IDF + Cosine Similarity on genres, cast, keywords, director)
- **Filters**: Industry (Bollywood / Hollywood), Genre, Rating, Year Range, Runtime, Actor search
- **Analytics tab**: Genre distribution, movies per year, rating histogram, industry split, top directors
- **Sober dark-gold UI** with Playfair Display typography

---

## ▶ Run Locally

```bash
# 1. Clone / unzip the project
cd movie_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (one-time)
python download_data.py

# 4. Launch
streamlit run app.py
```

---

## 🚀 Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a **public GitHub repo**  
   (make sure `data/` is committed, or add `download_data.py` to `requirements.txt`'s pre-run hook)

2. Go to → **https://share.streamlit.io**  
   Sign in with GitHub → **"New app"**

3. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `your-username/movie_recommender` |
   | Branch | `main` |
   | Main file path | `app.py` |

4. Click **Deploy!** — it will auto-install from `requirements.txt`.

> **Tip:** If you don't want to commit the CSVs, add a `startup.sh` or use  
> `st.cache_data` to download them at first run (see `download_data.py`).

---

## Dataset
- **TMDB 5000 Movies** — 4,803 films with budgets, genres, keywords, overview, ratings, runtime  
- **TMDB 5000 Credits** — cast & crew for all films  
- Source: [Kaggle TMDB Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Tech Stack
| Library | Use |
|---|---|
| Streamlit | UI & deployment |
| scikit-learn | TF-IDF vectorisation + cosine similarity |
| pandas / numpy | Data wrangling |
| altair | Charts & analytics |
