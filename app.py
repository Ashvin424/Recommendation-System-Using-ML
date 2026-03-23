import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%); 
        color: #f8fafc; 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    
    .main-title { 
        font-size: 3.5rem; font-weight: 800; text-align: center; margin-top: -30px; 
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
        margin-bottom: 5px;
    }
    
    .subtitle { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 40px; }

    .movie-card { 
        background: rgba(255, 255, 255, 0.04); 
        border: 1px solid rgba(255, 255, 255, 0.1); 
        border-radius: 20px; 
        padding: 25px; 
        
        display: flex; 
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        
        /* Fixed height to ensure uniformity across different title lengths */
        height: 280px; 
        width: 100%;
        box-sizing: border-box;
        
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }
    
    .movie-card:hover { 
        background: rgba(255, 255, 255, 0.07); 
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }
    
    .movie-name { 
        font-weight: 700; 
        font-size: 1.15rem; 
        color: #ffffff; 
        line-height: 1.4;
        text-align: center;
        
        /* Centering long/short titles vertically within the top space */
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 15px;
    }
   
    .imdb-link { text-decoration: none !important; width: 100%; display: flex; justify-content: center; }
    
    .imdb-btn {
        display: inline-block;
        width: 80%;
        background: #6366f1;
        color: white !important;
        padding: 10px 0;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 700;
        text-align: center;
        transition: background 0.2s;
        border: none;
    }

    .imdb-btn:hover { background: #4f46e5; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4); }

    div[data-testid="stVerticalBlock"] > div:has(div.movie-card) { padding: 0px !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movies.csv")
        df['genres_cleaned'] = df['genres'].str.replace('|', ' ', regex=False)
        return df
    except:
        return pd.DataFrame(columns=['title', 'genres', 'genres_cleaned'])

movies = load_data()

@st.cache_resource
def build_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data['genres_cleaned'])
    sim = cosine_similarity(matrix)
    return tfidf, matrix, sim

tfidf, tfidf_matrix, similarity = build_model(movies)

def get_recommendations(movie_name, top_n=6):
    titles = movies['title'].tolist()
    match = process.extractOne(movie_name, titles, scorer=fuzz.WRatio)
    if match and match[1] >= 60:
        idx = movies[movies['title'] == match[0]].index[0]
        sim_scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)
        return match[0], [movies.iloc[i[0]]['title'] for i in sim_scores[1:top_n+1]]
    return None, []

def get_personalized_recs(selected_genres, fav_movies, top_n=9):
    taste_profile = " ".join(selected_genres)
    for title in fav_movies:
        matches = movies[movies['title'] == title]
        if not matches.empty:
            m_genres = matches['genres_cleaned'].values[0]
            taste_profile += f" {m_genres}"
    
    user_vec = tfidf.transform([taste_profile])
    user_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    indices = user_sim.argsort()[::-1]
    
    results = []
    for i in indices:
        t = movies.iloc[i]['title']
        if t not in fav_movies and len(results) < top_n:
            results.append(t)
    return results

with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>Navigation</h2>", unsafe_allow_html=True)
    page = st.radio("Choose Mode:", ["Personalized Discovery", "Title Search", "Genre Explorer"])
    st.markdown("---")
    st.caption("Developed by Ashvin Prajapati")

def render_movie_cards(movie_list):
    # Standardized grid logic for all pages
    for i in range(0, len(movie_list), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(movie_list):
                movie = movie_list[i + j]
                url = f"https://www.imdb.com/find?q={movie.replace(' ', '+')}"
                with cols[j]:
                    st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-name">{movie}</div>
                            <a href="{url}" target="_blank" class="imdb-link">
                                <div class="imdb-btn">Details</div>
                            </a>
                        </div>
                    """, unsafe_allow_html=True)

if page == "Personalized Discovery":
    st.markdown("<div class='main-title'>Tailored Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Find movies based on your taste profile</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        u_genres = st.multiselect("Favorite Genres:", ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"])
    with c2:
        u_favs = st.multiselect("Movies You Liked:", movies['title'].unique())
    
    if st.button("Generate Dashboard"):
        if u_genres or u_favs:
            render_movie_cards(get_personalized_recs(u_genres, u_favs))
        else: 
            st.warning("Please add preferences first.")

elif page == "Title Search":
    st.markdown("<div class='main-title'>Smart Search</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        input_movie = st.text_input("", placeholder="Type a movie title...")
        search_trigger = st.button("Find Similar")
        
    if search_trigger and input_movie:
        match, recs = get_recommendations(input_movie)
        if match:
            st.markdown(f"<h4 style='text-align: center; color: #94a3b8;'>Results based on: {match}</h4>", unsafe_allow_html=True)
            render_movie_cards(recs)
        else: 
            st.error("Title not found.")

else:
    st.markdown("<div class='main-title'>Genre Explorer</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        sel_genre = st.selectbox("Select Genre", ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"])
        show_trigger = st.button("Show Movies")
        
    if show_trigger:
        results = movies[movies['genres'].str.contains(sel_genre, case=False)].head(9)['title'].tolist()
        render_movie_cards(results)