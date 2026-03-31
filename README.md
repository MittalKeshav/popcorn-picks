# 🍿 Popcorn Picks — Movie Recommendation System

A **content-based movie recommender web app** built with Streamlit, powered by the **TMDB 5000 Movie Dataset (~5,000 films)**.

🔗 **Live Demo:** https://popcorn-picks-k18m01.streamlit.app/

---

## 🚀 Features

* 🎯 **Smart Recommendations**
  Content-based filtering using **TF-IDF + Cosine Similarity** on:

  * Genres
  * Cast
  * Keywords
  * Director

* 🎛️ **Advanced Filtering System**

  * Industry (Bollywood / Hollywood)
  * Genre
  * Rating
  * Year Range
  * Language
  * Actor Search

* 📊 **Analytics Dashboard**

  * Genre distribution
  * Movies per year
  * Rating histogram
  * Industry split
  * Top directors

* 🎨 **Premium UI**

  * Dark cinematic theme
  * Gold-accent design
  * Interactive cards & posters
  * Smooth user experience

---

## 🧠 How It Works

1. Movies are converted into a **feature “soup”** combining:

   * genres + keywords + cast + director

2. Text is vectorized using **TF-IDF**

3. Similarity between movies is computed using:
   👉 **Cosine Similarity**

4. Based on user input, the system returns **most similar movies**

---

## ▶ Run Locally

```bash
# Clone repository
git clone https://github.com/your-username/popcorn-picks.git
cd popcorn-picks

# Install dependencies
pip install -r requirements.txt

# Download dataset (one-time)
python download_data.py

# Run app
streamlit run app.py
```

---

## 🌐 Deployment

Deployed using **Streamlit Community Cloud**

Steps:

1. Push code to GitHub
2. Connect repo on Streamlit Cloud
3. Select `app.py` → Deploy

---

## 📁 Dataset

* **TMDB 5000 Movies Dataset**
* ~4,800 movies with metadata (genres, ratings, overview, etc.)
* Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

---

## 🛠️ Tech Stack

| Technology     | Purpose                    |
| -------------- | -------------------------- |
| Streamlit      | UI & deployment            |
| scikit-learn   | TF-IDF & cosine similarity |
| pandas / numpy | Data processing            |
| Altair         | Data visualization         |
| OMDB API       | Movie posters & metadata   |

---

## ✨ Future Improvements

* 🔍 “Why this movie is recommended” explanation
* ⭐ User preference learning
* 📈 Trending / popular movies section
* 👤 User profiles

---

## 👨‍💻 Author

**Keshav Mittal**

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
