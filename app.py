import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    :root {
        --bg-main: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        --bg-sidebar: rgba(2, 6, 23, 0.92);
        --surface: rgba(255, 255, 255, 0.07);
        --surface-2: rgba(255, 255, 255, 0.10);
        --border: rgba(255, 255, 255, 0.14);
        --text-main: #f8fafc;
        --text-soft: #cbd5e1;
        --text-muted: #94a3b8;
        --accent-1: #6366f1;
        --accent-2: #a855f7;
        --accent-3: #ec4899;
        --success-bg: rgba(34, 197, 94, 0.15);
        --success-border: rgba(34, 197, 94, 0.35);
        --warning-bg: rgba(245, 158, 11, 0.15);
        --warning-border: rgba(245, 158, 11, 0.35);
        --error-bg: rgba(239, 68, 68, 0.15);
        --error-border: rgba(239, 68, 68, 0.35);
        --shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        --radius-xl: 22px;
        --radius-lg: 16px;
        --radius-md: 12px;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: var(--bg-main);
        color: var(--text-main);
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-sidebar);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-main) !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .main-title {
        font-size: 3.4rem;
        font-weight: 800;
        text-align: center;
        line-height: 1.1;
        margin-top: -0.5rem;
        margin-bottom: 0.35rem;
        background: linear-gradient(to right, var(--accent-1), var(--accent-2), var(--accent-3));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .section-heading {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-soft);
        margin-bottom: 0.75rem;
    }

    /* Labels */
    .stTextInput label,
    .stSelectbox label,
    .stMultiSelect label,
    .stRadio label,
    .stMarkdown,
    .stCaption,
    .stSubheader {
        color: var(--text-main) !important;
    }

    div[data-baseweb="input"] {
        background: rgba(15, 23, 42, 0.88) !important;
        border: 1px solid var(--border) !important;
        border-radius: 999px !important;
        min-height: 52px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="input"] input {
        color: var(--text-main) !important;
        background: transparent !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: var(--text-muted) !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.88) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        min-height: 52px !important;
        color: var(--text-main) !important;
    }

    div[data-baseweb="tag"] {
        background: rgba(99, 102, 241, 0.18) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        color: #e9eafe !important;
        border-radius: 999px !important;
    }

    .stButton > button {
        width: 100%;
        min-height: 48px;
        border: none;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
        color: white !important;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 10px 22px rgba(99, 102, 241, 0.25);
        transition: all 0.22s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(99, 102, 241, 0.35);
        filter: brightness(1.04);
    }

    .stButton > button:focus {
        outline: none !important;
        box-shadow: 0 0 0 0.2rem rgba(168, 85, 247, 0.25);
    }

    div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 0.35rem 0.6rem;
        margin-bottom: 0.35rem;
    }

    div[role="radiogroup"] label:hover {
        border-color: rgba(99, 102, 241, 0.35);
        background: rgba(255, 255, 255, 0.06);
    }

    .movie-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 22px 18px 18px 18px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        height: 270px;
        width: 100%;
        box-sizing: border-box;
        transition: all 0.28s ease;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
        margin-bottom: 18px;
    }

    .movie-card:hover {
        transform: translateY(-8px);
        background: var(--surface-2);
        border-color: rgba(99, 102, 241, 0.42);
        box-shadow: var(--shadow);
    }

    .movie-name {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--text-main);
        line-height: 1.45;
        text-align: center;
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        word-break: break-word;
    }

    .movie-meta {
        font-size: 0.86rem;
        color: var(--text-muted);
        margin-top: -0.25rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .imdb-link {
        text-decoration: none !important;
        width: 100%;
        display: flex;
        justify-content: center;
    }

    .imdb-btn {
        display: inline-block;
        width: 82%;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: #ffffff !important;
        padding: 10px 0;
        border-radius: 12px;
        font-size: 0.92rem;
        font-weight: 700;
        text-align: center;
        transition: all 0.22s ease;
        border: none;
    }

    .imdb-btn:hover {
        filter: brightness(1.05);
        box-shadow: 0 8px 18px rgba(99, 102, 241, 0.32);
    }

    [data-testid="stAlert"] {
        border-radius: 16px !important;
    }

    .stSuccess {
        background: var(--success-bg) !important;
        border: 1px solid var(--success-border) !important;
    }

    .stWarning {
        background: var(--warning-bg) !important;
        border: 1px solid var(--warning-border) !important;
    }

    .stError {
        background: var(--error-bg) !important;
        border: 1px solid var(--error-border) !important;
    }

    div[data-testid="stVerticalBlock"] > div:has(div.movie-card) {
        padding: 0 !important;
    }

    .hero-panel {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 1.2rem 1.2rem 0.5rem 1.2rem;
        margin-bottom: 1.4rem;
        backdrop-filter: blur(12px);
    }

    @media (max-width: 900px) {
        .main-title {
            font-size: 2.5rem;
        }

        .movie-card {
            height: 230px;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movies.csv")
        df["genres_cleaned"] = df["genres"].fillna("").str.replace("|", " ", regex=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["title", "genres", "genres_cleaned"])

movies = load_data()

@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(data["genres_cleaned"])
    sim = cosine_similarity(matrix)
    return tfidf, matrix, sim

tfidf, tfidf_matrix, similarity = build_model(movies)

def get_imdb_link(movie_title: str) -> str:
    query = movie_title.replace(" ", "+")
    return f"https://www.imdb.com/find?q={query}"

def get_recommendations(movie_name, top_n=6):
    titles = movies["title"].tolist()
    match = process.extractOne(movie_name, titles, scorer=fuzz.WRatio)

    if match and match[1] >= 60:
        idx = movies[movies["title"] == match[0]].index[0]
        sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
        recs = [movies.iloc[i[0]]["title"] for i in sim_scores[1:top_n+1]]
        return match[0], recs

    return None, []

def get_personalized_recs(selected_genres, fav_movies, top_n=9):
    taste_profile = " ".join(selected_genres)

    for title in fav_movies:
        matches = movies[movies["title"] == title]
        if not matches.empty:
            taste_profile += f" {matches['genres_cleaned'].values[0]}"

    user_vec = tfidf.transform([taste_profile])
    user_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    indices = user_sim.argsort()[::-1]

    results = []
    for i in indices:
        t = movies.iloc[i]["title"]
        if t not in fav_movies and len(results) < top_n:
            results.append(t)

    return results

def get_genre_recs(selected_genre, top_n=9):
    return (
        movies[movies["genres"].str.contains(selected_genre, case=False, na=False)]
        .head(top_n)["title"]
        .tolist()
    )

with st.sidebar:
    st.markdown("##  Navigation")
    page = st.radio(
        "Choose Mode:",
        ["Personalized Discovery", "Title Search", "Genre Explorer"]
    )
    st.markdown("---")
    st.caption("Developed by Ashvin Prajapati")

def render_movie_cards(movie_list):
    for i in range(0, len(movie_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(movie_list):
                movie = movie_list[i + j]
                url = get_imdb_link(movie)
                with cols[j]:
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div class="movie-name">{movie}</div>
                            <a href="{url}" target="_blank" class="imdb-link">
                                <div class="imdb-btn">View on IMDb</div>
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

if page == "Personalized Discovery":
    st.markdown("<div class='main-title'>Tailored Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Build a taste profile using genres and movies you already like</div>", unsafe_allow_html=True)

    st.markdown("<div class='hero-panel'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        u_genres = st.multiselect(
            "Favorite Genres",
            ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
             "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"]
        )

    with c2:
        u_favs = st.multiselect(
            "Movies You Liked",
            movies["title"].unique()
        )

    generate_btn = st.button("Generate Recommendations")
    st.markdown("</div>", unsafe_allow_html=True)

    if generate_btn:
        if u_genres or u_favs:
            render_movie_cards(get_personalized_recs(u_genres, u_favs))
        else:
            st.warning("Please select at least one genre or one favorite movie.")

elif page == "Title Search":
    st.markdown("<div class='main-title'>Smart Title Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Type any movie title and get similar recommendations instantly</div>", unsafe_allow_html=True)

    st.markdown("<div class='hero-panel'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        input_movie = st.text_input("", placeholder="Type a movie title...")
        search_trigger = st.button("Find Similar Movies")
    st.markdown("</div>", unsafe_allow_html=True)

    if search_trigger:
        if not input_movie.strip():
            st.warning("Please enter a movie title.")
        else:
            match, recs = get_recommendations(input_movie)
            if match:
                st.success(f"Showing recommendations based on: {match}")
                render_movie_cards(recs)
            else:
                st.error("Title not found. Try another movie name.")

else:
    st.markdown("<div class='main-title'>Genre Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Browse movies by genre and jump to IMDb for more details</div>", unsafe_allow_html=True)

    st.markdown("<div class='hero-panel'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        sel_genre = st.selectbox(
            "Select Genre",
            ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
             "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"]
        )
        show_trigger = st.button("Show Movies")
    st.markdown("</div>", unsafe_allow_html=True)

    if show_trigger:
        render_movie_cards(get_genre_recs(sel_genre))