import streamlit as st
import pandas as pd
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from tenacity import retry, stop_after_attempt, wait_exponential

# SETUP
st.set_page_config(
    page_title="🎬 Movie Recommender",
    layout="wide",
    page_icon="🎥",
    initial_sidebar_state="expanded"
)

# CSS ()
def css():
    st.markdown("""
    <style>
        .stApp { background-color: #0f0f1a; color: #ffffff; }
        .st-emotion-cache-10trblm { color: #ff4b4b; font-size: 2.5rem; }
        .stTextInput input {
            background-color: #1e1e2d !important;
            color: white !important;
            border: 1px solid #ff4b4b !important;
            border-radius: 8px !important;
        }
        .movie-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
            background: #1e1e2d;
            border: 1px solid #2a2a3a;
        }
        .movie-card:hover { transform: scale(1.05); box-shadow: 0 10px 20px rgba(255, 75, 75, 0.2); }
        .movie-title { font-weight: bold; color: white; padding: 10px; font-size: 1rem; }
        .movie-overview { color: #b3b3b3; padding: 0 10px 10px; font-size: 0.85rem; }
        .stButton>button { background-color: #ff4b4b !important; color: white !important; border-radius: 8px !important; }
        .error-poster { background-color: #2a2a3a; padding: 40% 0; text-align: center; color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)
css()

# DATA LOADING 
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")  
    movies = movies[['id', 'title', 'overview']].dropna()
    movies['title_lower'] = movies['title'].str.lower()
    return movies

movies = load_data()

# NLP MODEL
@st.cache_data
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  

model = load_model()

# Precompute embeddings for all movies
@st.cache_data
def get_embeddings():
    return model.encode(movies['overview'].tolist())

embeddings = get_embeddings()

# POSTER FETCHING ()
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c6b5a96e51b81671f2364510d1040e29&language=en-US"
    response = requests.get(url, timeout=5)
    if response.ok:
        return f"https://image.tmdb.org/t/p/w500/{response.json().get('poster_path')}"
    return "https://via.placeholder.com/500x750?text=No+Poster"

# RECOMMENDATION LOGIC 
def recommend_by_plot(user_plot):

    user_embedding = model.encode([user_plot])[0]
    sim_scores = cosine_similarity([user_embedding], embeddings)[0]
    
    # Get top 5 matches
    top_indices = np.argsort(sim_scores)[-5:][::-1]
    recommendations = movies.iloc[top_indices].copy()
    recommendations['poster'] = recommendations['id'].apply(fetch_poster)
    
    return None, recommendations 

def recommend_by_title(movie_title):
    # Fuzzy match title
    matches = difflib.get_close_matches(movie_title.lower(), movies['title_lower'], n=1, cutoff=0.6)
    if not matches:
        return None, None
    
    # Find closest match
    idx = movies[movies['title_lower'] == matches[0]].index[0]
    
    # Get similar movies
    sim_scores = cosine_similarity([embeddings[idx]], embeddings)[0]
    top_indices = np.argsort(sim_scores)[-6:][::-1]  
    recommended_indices = [i for i in top_indices if i != idx][:5] 
    
    recommendations = movies.iloc[recommended_indices].copy()
    recommendations['poster'] = recommendations['id'].apply(fetch_poster)
    
    return movies.loc[idx, 'title'], recommendations

# STREAMLIT UI
st.title("🍿 Movie Recommender System")
st.write("Movie recommender based on storyline")
st.markdown("---")
movie_input = st.text_input("🔍 Search for a movie:", key="search_input", placeholder="Search any movie story line ...")
if movie_input and (st.session_state.get("search_input")):
    with st.spinner('Finding similar stories...'):
        # Check if input matches a movie title
        title_matches = difflib.get_close_matches(movie_input.lower(), movies['title_lower'], n=1, cutoff=0.6)
        
        if title_matches:
            matched_title, recommendations = recommend_by_title(movie_input)
            if matched_title:
                st.success(f"Because you liked **{matched_title}**, you might enjoy:")
        else:
            matched_title, recommendations = recommend_by_plot(movie_input)
            st.success("Recommended movies based on your story description:")
            st.markdown("---")
        
        # Display results
        if recommendations is not None:
            cols = st.columns(5)
            for idx, row in enumerate(recommendations.itertuples()):
                with cols[idx % 5]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <img src="{row.poster}" width="100%">
                        <div style="padding: 10px;">
                            <strong>{row.title}</strong>
                            <p style="color: #aaa; font-size: 0.8em;">{row.overview[:80]}...</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("More info"):
                            st.write(f"**Overview:** {row.overview}")
                            st.write(f"**Movie ID:** {row.id}")
        else:
            st.error("No matches found. Try another query!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #b3b3b3; font-size: 0.8rem;">
    <p>Powered by TMDb • Recommendations based on movie overviews</p>
    <p>© 2025 Movie Recommender - Your Personal Movie Guide</p>
</div>
""", unsafe_allow_html=True)