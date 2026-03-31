import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import requests
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Popcorn Picks – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (plain string — no f-string — so curly braces are safe)
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0d0d;
    --surface: #161616;
    --card:    #1e1e1e;
    --border:  #2a2a2a;
    --gold:    #c9a84c;
    --gdim:    #7a6230;
    --text:    #e8e2d6;
    --muted:   #7a7570;
    --accent:  #e8a44a;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 4rem; font-weight: 900; letter-spacing: -1px;
    background: linear-gradient(135deg, #c9a84c 0%, #f5d98b 50%, #c9a84c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin: 0;
    text-align: center;
}
.hero-sub {
    font-size: 1rem; color: var(--muted);
    letter-spacing: 3px; text-transform: uppercase; margin-top: 6px;
    text-align: center;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gdim), transparent);
    margin: 1.5rem 0;
}

.movie-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}
.movie-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--gdim), var(--gold));
    opacity: 0; transition: opacity 0.2s;
}
.movie-card:hover { border-color: var(--gdim); transform: translateY(-3px); }
.movie-card:hover::before { opacity: 1; }

.movie-meta {
    font-size: 0.92rem; color: var(--muted); margin-bottom: 8px;
    display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
}
.badge {
    background: var(--border); border-radius: 4px; padding: 2px 9px;
    font-size: 0.82rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-hwood { background: rgba(201,168,76,0.15); color: #c9a84c; border: 1px solid #7a6230; }
.badge-bwood { background: rgba(232,164,74,0.15); color: #e8a44a; border: 1px solid rgba(232,164,74,0.4); }
.star { color: var(--gold); font-size: 1rem; }

.poster {
    border-radius: 8px; height: 210px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin-bottom: 10px; position: relative; overflow: hidden;
}
.poster-initial {
    font-family: 'Playfair Display', serif; font-size: 3.6rem; font-weight: 900;
    color: rgba(255,255,255,0.9); text-shadow: 0 2px 12px rgba(0,0,0,0.5);
    letter-spacing: -1px; z-index: 1;
}
.poster-year {
    position: absolute; bottom: 8px; right: 10px;
    font-size: 0.68rem; color: rgba(255,255,255,0.55); letter-spacing: 1px; z-index: 1;
}

.stat-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 700; color: var(--gold);
}
.stat-label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

/* ── Sidebar labels & section headers ── */
.sidebar-section {
    font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase;
    color: var(--gold); margin: 1.2rem 0 0.5rem;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text) !important; font-size: 0.88rem !important;
}

/* ── Placeholder text — force visible on ALL inputs ── */
input::placeholder,
textarea::placeholder,
[data-testid="stTextInput"] input::placeholder,
[data-testid="stSidebar"] input::placeholder,
[data-baseweb="input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder,
::-webkit-input-placeholder,
[data-testid="stSidebar"] ::-webkit-input-placeholder,
[data-testid="stTextInput"] ::-webkit-input-placeholder {
    color: #7a7570 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #7a7570 !important;
}

/* ── All sidebar input widgets: dark background, light text ── */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] input::placeholder { color: var(--muted) !important; opacity: 1 !important; }

/* ── Selectbox & multiselect: dark background + VISIBLE text ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div,
[data-baseweb="popover"] {
    background: var(--card) !important;
    border-color: var(--border) !important;
}

/* The selected value text (e.g. "All", "Weighted Rating", "Choose options") */
[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="value"],
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="placeholder"],
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="singleValue"],
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] p,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] div,
[data-testid="stSidebar"] [data-testid="stMultiSelect"] span {
    color: #c9a84c !important;
    -webkit-text-fill-color: #c9a84c !important;
}

/* Tags inside multiselect (e.g. Hollywood x, Bollywood x) */
[data-testid="stSidebar"] [data-baseweb="tag"],
[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #e8e2d6 !important;
    -webkit-text-fill-color: #e8e2d6 !important;
}

/* Dropdown option list */
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="popover"] li {
    background: var(--card) !important;
    color: #c9a84c !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] li:hover {
    background: var(--border) !important;
}

/* ── Main area search box ── */
[data-testid="stTextInput"] input {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 8px !important;
    font-size: 0.97rem !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; opacity: 1 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: var(--gdim) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #b8922a, #c9a84c) !important;
    color: #0d0d0d !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important; transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; margin-top: 4px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important; font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important; font-weight: 700 !important; padding: 0.6rem 0.8rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--gold) !important;
    background: rgba(201,168,76,0.06) !important; border-radius: 8px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: var(--muted) !important; font-size: 1.05rem !important;
    font-weight: 500 !important; background: transparent !important; border: none !important;
    padding: 0.5rem 1.2rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""
<style>

/* 🔥 FIX: remove black overlay from multiselect tags */
[data-baseweb="tag"] {
    position: relative !important;
    z-index: 10 !important;
}

/* remove internal dark layers */
[data-baseweb="tag"] * {
    background: transparent !important;
    box-shadow: none !important;
}

/* kill pseudo elements */
[data-baseweb="tag"]::before,
[data-baseweb="tag"]::after {
    display: none !important;
}

/* keep text above */
[data-baseweb="tag"] span {
    position: relative !important;
    z-index: 20 !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
OMDB_API_KEY = "1d739620"

GENRE_GRADIENTS = {
    "Action":          "linear-gradient(135deg,#1a1a2e,#16213e,#e05c5c)",
    "Adventure":       "linear-gradient(135deg,#0f3460,#16213e,#e94560)",
    "Animation":       "linear-gradient(135deg,#2d6a4f,#1b4332,#d4a017)",
    "Comedy":          "linear-gradient(135deg,#f4a261,#e76f51,#264653)",
    "Crime":           "linear-gradient(135deg,#1c1c1c,#2d2d2d,#c9a84c)",
    "Documentary":     "linear-gradient(135deg,#2b4162,#12100e,#4a6fa5)",
    "Drama":           "linear-gradient(135deg,#2c3e50,#3d2b1f,#c9a84c)",
    "Fantasy":         "linear-gradient(135deg,#1a0533,#3d1a78,#9b59b6)",
    "Horror":          "linear-gradient(135deg,#0a0a0a,#1a0000,#8b0000)",
    "Romance":         "linear-gradient(135deg,#3d0c11,#6b2737,#d4a5a5)",
    "Sci-Fi":          "linear-gradient(135deg,#0d0d2b,#1a1a4e,#00b4d8)",
    "Science Fiction": "linear-gradient(135deg,#0d0d2b,#1a1a4e,#00b4d8)",
    "Thriller":        "linear-gradient(135deg,#1c1c2e,#2d1b3d,#6a0572)",
    "History":         "linear-gradient(135deg,#3e2723,#5d4037,#c9a84c)",
    "Music":           "linear-gradient(135deg,#1a237e,#283593,#7986cb)",
    "War":             "linear-gradient(135deg,#212121,#424242,#78909c)",
    "Western":         "linear-gradient(135deg,#3e2000,#6d3a00,#c9a84c)",
    "Family":          "linear-gradient(135deg,#1b5e20,#2e7d32,#a5d6a7)",
}
DEFAULT_GRADIENT = "linear-gradient(135deg,#1a1a2e,#16213e,#2d2d2d)"

LANG_MAP = {
    "English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es",
    "German": "de", "Japanese": "ja", "Korean": "ko", "Italian": "it",
    "Tamil": "ta", "Telugu": "te", "Chinese": "zh",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    base     = os.path.dirname(os.path.abspath(__file__))
    movies   = pd.read_csv(os.path.join(base, "data", "tmdb_5000_movies.csv"))
    credits  = pd.read_csv(os.path.join(base, "data", "tmdb_5000_credits.csv"))
    credits.rename(columns={"movie_id": "id"}, inplace=True)
    df = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))

    def safe_parse(v):
        try:    return ast.literal_eval(v)
        except: return []

    def extract_names(v, key="name", limit=None):
        items = safe_parse(v) if isinstance(v, str) else (v or [])
        names = [i[key] for i in items if key in i]
        return names[:limit] if limit else names

    df["genres_list"]   = df["genres"].apply(extract_names)
    df["keywords_list"] = df["keywords"].apply(extract_names)
    df["cast_list"]     = df["cast"].apply(lambda x: extract_names(x, limit=5))
    df["director"]      = df["crew"].apply(lambda x: next(
        (m["name"] for m in safe_parse(x) if isinstance(m, dict) and m.get("job") == "Director"),
        "Unknown"))

    def get_industry(row):
        lang  = str(row.get("original_language", "")).lower()
        cntry = str(row.get("production_countries", "")).lower()
        if lang in ("hi","te","ta","ml","kn","bn","mr","pa") or "india" in cntry:
            return "Bollywood"
        return "Hollywood"

    df["industry"]     = df.apply(get_industry, axis=1)
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    def make_soup(row):
        parts = row["genres_list"] + row["keywords_list"] + row["cast_list"] + [row["director"]] * 2
        return " ".join(p.lower().replace(" ", "") for p in parts)

    df["soup"] = df.apply(make_soup, axis=1)

    C = df["vote_average"].median()
    m = df["vote_count"].quantile(0.70)
    df["weighted_rating"] = df.apply(
        lambda r: (r["vote_count"] / (r["vote_count"] + m)) * r["vote_average"] +
                  (m / (r["vote_count"] + m)) * C, axis=1)

    title_col = "title_x" if "title_x" in df.columns else "title"
    df["title"] = df[title_col]
    df = df[df["title"].notna()].reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_similarity(df):
    tfidf  = TfidfVectorizer(stop_words="english", max_features=15000)
    matrix = tfidf.fit_transform(df["soup"].fillna(""))
    return cosine_similarity(matrix, matrix)


def get_recommendations(df, sim, title, n=12):
    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        matches = df[df["title"].str.lower().str.startswith(title.lower()[:4])]
    if matches.empty:
        return pd.DataFrame()
    idx    = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    return df.iloc[[i for i, _ in scores]].copy()


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_omdb(title: str, year: int):
    try:
        params = {"apikey": OMDB_API_KEY, "t": title, "type": "movie"}
        if year and year > 0:
            params["y"] = year
        data = requests.get("https://www.omdbapi.com/", params=params, timeout=5).json()
        if data.get("Response") == "True":
            poster  = data.get("Poster", "")
            imdb_id = data.get("imdbID", "")
            return (
                poster if poster and poster != "N/A" else "",
                "https://www.imdb.com/title/" + imdb_id + "/" if imdb_id else "",
            )
    except Exception:
        pass
    return "", ""


def star_str(rating):
    full = int(round(rating / 2))
    return "★" * full + "☆" * (5 - full)


# ─────────────────────────────────────────────────────────────────────────────
# Card
# ─────────────────────────────────────────────────────────────────────────────
def render_card(row, idx):
    genres     = ", ".join(row["genres_list"][:3]) if row["genres_list"] else "—"
    all_genres = ", ".join(row["genres_list"][:5]) if row["genres_list"] else "—"
    year       = int(row["release_year"]) if row["release_year"] else 0
    year_disp  = str(year) if year else "—"
    rating     = round(row["vote_average"], 1)
    ind        = row["industry"]
    badge_cls  = "badge-hwood" if ind == "Hollywood" else "badge-bwood"
    gradient   = GENRE_GRADIENTS.get(row["genres_list"][0] if row["genres_list"] else "", DEFAULT_GRADIENT)
    words      = str(row["title"]).split()
    initials   = (words[0][0] + words[1][0]).upper() if len(words) >= 2 else str(row["title"])[:2].upper()
    stars      = star_str(rating)

    poster_url, imdb_url = fetch_omdb(row["title"], year)
    if not imdb_url:
        imdb_url = "https://www.imdb.com/find?q=" + str(row["title"]).replace(" ", "+") + "&s=tt&ttype=ft"

    if poster_url:
        poster_html = (
            '<img src="' + poster_url + '" style="width:100%;height:220px;'
            'object-fit:cover;border-radius:8px;margin-bottom:10px;" loading="lazy"/>'
        )
    else:
        poster_html = (
            '<div class="poster" style="background:' + gradient + ';">'
            '<div class="poster-initial">' + initials + '</div>'
            '<div class="poster-year">' + year_disp + '</div>'
            '</div>'
        )

    st.markdown(
        '<div class="movie-card">'
        + poster_html +
        '<div class="movie-meta">'
        '<span class="star">' + stars + '</span>'
        '<span style="font-size:0.95rem;">' + str(rating) + '/10</span>'
        '<span class="badge ' + badge_cls + '">' + ind + '</span>'
        '</div>'
        '<div class="movie-meta"><span class="badge">' + genres + '</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    overview = str(row.get("overview", "No overview available."))
    cast     = ", ".join(row["cast_list"][:5]) if row["cast_list"] else "—"
    director = str(row.get("director", "—"))

    with st.expander("📋  " + str(row["title"])):
        st.markdown("## [" + str(row["title"]) + " (" + year_disp + ")](" + imdb_url + ")")
        st.caption(stars + "  " + str(rating) + "/10  |  " + ind + "  |  " + all_genres)
        st.write(overview)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("🎭 LEAD CAST")
            st.write(cast)
        with c2:
            st.caption("🎬 DIRECTOR")
            st.write(director)

        if "watchlist" not in st.session_state:
            st.session_state["watchlist"] = []
        already = any(w["title"] == row["title"] for w in st.session_state["watchlist"])
        if already:
            st.success("✓ Already in your Watchlist")
        else:
            if st.button("❤️ Add to Watchlist", key="wl_" + str(idx)):
                st.session_state["watchlist"].append({
                    "title":      row["title"],
                    "year":       year_disp,
                    "rating":     rating,
                    "industry":   ind,
                    "genres":     all_genres,
                    "cast":       cast,
                    "director":   director,
                    "overview":   overview,
                    "imdb_url":   imdb_url,
                    "poster_url": poster_url,
                })
                st.rerun()


def render_grid(data, prefix, cols=3):
    data = data.reset_index(drop=True)
    for row_start in range(0, len(data), cols):
        columns = st.columns(cols)
        for ci, di in enumerate(range(row_start, min(row_start + cols, len(data)))):
            with columns[ci]:
                render_card(data.iloc[di], prefix + "_" + str(di))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.spinner("Loading cinematic database…"):
        df  = load_data()
        sim = build_similarity(df)

    st.markdown(
        '<div style="padding:2rem 0 1rem;text-align:center;">'
        '<div class="hero-title">🍿 Popcorn Picks</div>'
        '<div class="hero-sub">Your personal cinema universe</div>'
        '</div>'
        '<div class="divider"></div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in [
        (c1, str(len(df)) + " total",                                          "Movies"),
        (c2, str(df["industry"].value_counts().get("Bollywood", 0)),           "Bollywood"),
        (c3, str(df["industry"].value_counts().get("Hollywood", 0)),           "Hollywood"),
        (c4, str(df["release_year"].nunique()),                                "Years Covered"),
    ]:
        col.markdown(
            '<div class="stat-box">'
            '<div class="stat-num">' + num + '</div>'
            '<div class="stat-label">' + label + '</div>'
            '</div>',
            unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(
            '<div style="font-family:\'Playfair Display\',serif;font-size:1.4rem;'
            'font-weight:700;color:#c9a84c;margin-bottom:4px;">🎛️ Filters</div>'
            '<hr style="border-color:#2a2a2a;margin-bottom:1rem;">',
            unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Industry</div>', unsafe_allow_html=True)
        industry_filter = st.multiselect("Industry", ["Hollywood", "Bollywood"],
            default=["Hollywood", "Bollywood"], label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">Genre</div>', unsafe_allow_html=True)
        all_genre_opts = sorted({g for lst in df["genres_list"] for g in lst})
        genre_filter = st.multiselect("Genre", all_genre_opts, label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">Min Rating</div>', unsafe_allow_html=True)
        min_rating = st.slider("Min Rating", 0.0, 10.0, 5.0, 0.5, label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">Year Range</div>', unsafe_allow_html=True)
        yr_min = int(df[df["release_year"] > 0]["release_year"].min())
        yr_max = int(df["release_year"].max())
        year_range = st.slider("Year Range", yr_min, yr_max, (1990, yr_max), label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">Language</div>', unsafe_allow_html=True)
        lang_filter = st.selectbox("Language", ["All"] + sorted(LANG_MAP.keys()), label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">Actor Search</div>', unsafe_allow_html=True)
        actor_search = st.text_input("Actor", placeholder="e.g. Tom Hanks", label_visibility="collapsed")

        st.markdown('<div class="sidebar-section">Sort By</div>', unsafe_allow_html=True)
        sort_by = st.selectbox("Sort By",
            ["Weighted Rating", "Release Year", "Popularity", "Vote Count"],
            label_visibility="collapsed")

    # Filters
    filtered = df.copy()
    if industry_filter:
        filtered = filtered[filtered["industry"].isin(industry_filter)]
    if genre_filter:
        filtered = filtered[filtered["genres_list"].apply(lambda g: any(x in g for x in genre_filter))]
    filtered = filtered[filtered["vote_average"] >= min_rating]
    filtered = filtered[filtered["release_year"].between(*year_range)]
    if lang_filter != "All":
        filtered = filtered[filtered["original_language"] == LANG_MAP[lang_filter]]
    if actor_search.strip():
        q = actor_search.strip().lower()
        filtered = filtered[filtered["cast_list"].apply(lambda c: any(q in a.lower() for a in c))]

    sort_col, sort_asc = {
        "Weighted Rating": ("weighted_rating", False),
        "Release Year":    ("release_year",     False),
        "Popularity":      ("popularity",        False),
        "Vote Count":      ("vote_count",        False),
    }[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=sort_asc)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Browse & Discover", "🔍 Get Recommendations", "📊 Insights", "❤️ Watchlist"
    ])

    # Tab 1
    with tab1:
        title_search = st.text_input("search",
            placeholder="🔎  Search by title e.g. Inception, Sholay…",
            label_visibility="collapsed")
        display_df = filtered[
            filtered["title"].str.contains(title_search.strip(), case=False, na=False)
        ] if title_search.strip() else filtered

        st.markdown(
            "<p style='color:#7a7570;font-size:0.92rem;margin-bottom:1rem;'>"
            "Showing <b style='color:#c9a84c'>" + str(min(24, len(display_df))) + "</b> of "
            "<b style='color:#e8e2d6'>" + str(len(display_df)) + "</b> movies</p>",
            unsafe_allow_html=True)

        if display_df.empty:
            st.info("No movies found. Try a different search or adjust filters.")
        else:
            render_grid(display_df.head(24), "t1")

    # Tab 2
    with tab2:
        st.markdown("<p style='color:#7a7570;font-size:0.95rem;margin-bottom:1rem;'>"
                    "Enter a movie you love and we'll find similar films.</p>",
                    unsafe_allow_html=True)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            query_title = st.text_input("Movie title", placeholder="e.g. The Dark Knight",
                                        label_visibility="collapsed")
        with col_b:
            n_recs = st.select_slider("How many?", [6, 12, 18], value=12)

        if st.button("✨ Find Similar Movies"):
            if not query_title.strip():
                st.warning("Please enter a movie title.")
            else:
                recs = get_recommendations(df, sim, query_title.strip(), n=n_recs)
                if recs.empty:
                    st.error("Could not find '" + query_title + "'. Try another title.")
                else:
                    if industry_filter:
                        recs = recs[recs["industry"].isin(industry_filter)]
                    if genre_filter:
                        recs = recs[recs["genres_list"].apply(lambda g: any(x in g for x in genre_filter))]
                    recs = recs[recs["vote_average"] >= min_rating]
                    st.markdown(
                        "<p style='color:#7a7570;font-size:0.92rem;'>Found "
                        "<b style='color:#c9a84c'>" + str(len(recs)) + "</b> recommendations for "
                        "<b style='color:#e8e2d6'>" + query_title + "</b></p>",
                        unsafe_allow_html=True)
                    render_grid(recs, "t2")

        st.markdown("<div class='divider'></div><p style='color:#7a7570;font-size:0.88rem;'>Try these:</p>",
                    unsafe_allow_html=True)
        seeds = ["The Dark Knight", "Inception", "Interstellar", "The Godfather",
                 "Avengers", "Dilwale Dulhania Le Jayenge", "3 Idiots"]
        seed_cols = st.columns(len(seeds))
        for sc, seed in zip(seed_cols, seeds):
            if sc.button(seed, key="seed_" + seed):
                recs = get_recommendations(df, sim, seed, n=12)
                if not recs.empty:
                    st.session_state["rec_title"]   = seed
                    st.session_state["rec_results"] = recs

        if "rec_results" in st.session_state:
            st.markdown(
                "<p style='color:#7a7570;font-size:0.92rem;margin-top:1rem;'>Similar to "
                "<b style='color:#e8e2d6'>" + st.session_state["rec_title"] + "</b>:</p>",
                unsafe_allow_html=True)
            render_grid(st.session_state["rec_results"], "t2s")

    # Tab 3
    with tab3:
        import altair as alt
        from collections import Counter

        st.markdown("<p style='color:#7a7570;margin-bottom:1.5rem;'>Analytics on filtered dataset.</p>",
                    unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("<p style='color:#c9a84c;font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;'>Top Genres</p>",
                        unsafe_allow_html=True)
            gc    = Counter(g for lst in filtered["genres_list"] for g in lst)
            gc_df = pd.DataFrame(gc.most_common(12), columns=["genre", "count"])
            st.altair_chart(
                alt.Chart(gc_df)
                .mark_bar(color="#c9a84c", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(x=alt.X("count:Q", title=""), y=alt.Y("genre:N", sort="-x", title=""))
                .properties(height=300, background="transparent"),
                use_container_width=True)

        with r1c2:
            st.markdown("<p style='color:#c9a84c;font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;'>Movies Per Year</p>",
                        unsafe_allow_html=True)
            yr_df = (filtered[filtered["release_year"] > 1900]
                     .groupby("release_year").size().reset_index(name="count"))
            st.altair_chart(
                alt.Chart(yr_df)
                .mark_area(color="#c9a84c", opacity=0.6, line={"color": "#c9a84c"})
                .encode(x=alt.X("release_year:O", title=""), y=alt.Y("count:Q", title=""))
                .properties(height=300, background="transparent"),
                use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("<p style='color:#c9a84c;font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;'>Rating Distribution</p>",
                        unsafe_allow_html=True)
            hdf = pd.cut(filtered["vote_average"].dropna(), bins=20).value_counts().reset_index()
            hdf.columns = ["bin", "count"]
            hdf["bin"] = hdf["bin"].astype(str)
            hdf = hdf.sort_values("bin")
            st.altair_chart(
                alt.Chart(hdf)
                .mark_bar(color="#e8a44a", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(x=alt.X("bin:O", title="Rating"), y=alt.Y("count:Q", title=""))
                .properties(height=280, background="transparent"),
                use_container_width=True)

        with r2c2:
            st.markdown("<p style='color:#c9a84c;font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;'>Industry Split</p>",
                        unsafe_allow_html=True)
            idf = filtered["industry"].value_counts().reset_index()
            idf.columns = ["industry", "count"]
            st.altair_chart(
                alt.Chart(idf).mark_arc(innerRadius=60, outerRadius=110)
                .encode(
                    theta="count:Q",
                    color=alt.Color("industry:N",
                        scale=alt.Scale(domain=["Hollywood", "Bollywood"],
                                        range=["#c9a84c", "#e8a44a"])),
                    tooltip=["industry", "count"])
                .properties(height=280, background="transparent"),
                use_container_width=True)

        st.markdown(
            "<div class='divider'></div>"
            "<p style='color:#c9a84c;font-size:0.85rem;letter-spacing:1px;text-transform:uppercase;'>"
            "Top Directors (by Avg Rating)</p>",
            unsafe_allow_html=True)
        dir_df = (filtered.groupby("director")
                  .agg(avg_rating=("weighted_rating", "mean"), movie_count=("title", "count"))
                  .query("movie_count >= 3")
                  .sort_values("avg_rating", ascending=False)
                  .head(10).reset_index())
        st.altair_chart(
            alt.Chart(dir_df)
            .mark_bar(color="#c9a84c", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("avg_rating:Q", title="Avg Weighted Rating"),
                y=alt.Y("director:N", sort="-x", title=""),
                tooltip=["director", "avg_rating", "movie_count"])
            .properties(height=320, background="transparent"),
            use_container_width=True)

    # Tab 4
    with tab4:
        wl = st.session_state.get("watchlist", [])
        if not wl:
            st.markdown(
                "<div style='text-align:center;padding:4rem 0;'>"
                "<div style='font-size:3rem;margin-bottom:1rem;'>🎬</div>"
                "<div style='font-size:1.4rem;color:#94a3b8;font-family:\"Playfair Display\",serif;"
                "margin-bottom:0.5rem;'>Your watchlist is empty</div>"
                "<div style='font-size:0.9rem;color:#7a7570;'>Open any movie and click ❤️ Add to Watchlist</div>"
                "</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<p style='color:#7a7570;font-size:0.92rem;margin-bottom:1rem;'>"
                "<b style='color:#c9a84c'>" + str(len(wl)) + "</b> movies saved</p>",
                unsafe_allow_html=True)
            if st.button("🗑️ Clear Watchlist"):
                st.session_state["watchlist"] = []
                st.rerun()
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            for i, m in enumerate(wl):
                stars = "★" * int(round(m["rating"] / 2)) + "☆" * (5 - int(round(m["rating"] / 2)))
                with st.expander("📋  " + m["title"] + "  (" + str(m["year"]) + ")  —  " + stars + "  " + str(m["rating"]) + "/10"):
                    if m.get("poster_url"):
                        st.image(m["poster_url"], width=160)
                    st.markdown("## [" + m["title"] + " (" + str(m["year"]) + ")](" + m["imdb_url"] + ")")
                    st.caption(stars + "  " + str(m["rating"]) + "/10  |  " + m["industry"] + "  |  " + m["genres"])
                    st.write(m["overview"])
                    wc1, wc2 = st.columns(2)
                    with wc1:
                        st.caption("🎭 LEAD CAST")
                        st.write(m["cast"])
                    with wc2:
                        st.caption("🎬 DIRECTOR")
                        st.write(m["director"])
                    if st.button("🗑️ Remove", key="wl_rm_" + str(i)):
                        st.session_state["watchlist"].pop(i)
                        st.rerun()


if __name__ == "__main__":
    main()