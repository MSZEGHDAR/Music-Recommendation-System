import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from joblib import load
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')

st.set_page_config(page_title="Spotify Song Recommender", page_icon=":musical_note:", layout="wide")

st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }
    .song-container {
        margin-bottom: 20px;
        background-color: #181818;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .song-content {
        display: flex;
        align-items: center;
    }
    .song-info {
        padding: 15px;
        flex-grow: 1;
    }
    .song-title {
        font-size: 24px !important;
        font-weight: bold;
        color: #ffffff;
        margin: 0;
    }
    .song-details {
        margin: 8px 0;
        color: #b3b3b3;
        font-size: 14px;
    }
    .detail-label {
        color: #1DB954;
    }
    .stTextInput > div > div > input {
        width: 100%;
    }
    .stSelectbox > div > div > select {
        width: 100%;
    }
    h1, h2 {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')
    prepared_data = load('prepared_data.joblib')
    preprocessed_data = load('preprocessed_data.joblib')
    nn_model = load('nn_model.joblib')
    return data, prepared_data, preprocessed_data, nn_model

data, prepared_data, preprocessed_data, nn_model = load_data()

def get_recommendations(nn_model, dataset, preprocessed_data, seed_song_id, n_recommendations=20):
    seed_index = dataset[dataset['id'] == seed_song_id].index[0]
    distances, indices = nn_model.kneighbors(preprocessed_data[seed_index].reshape(1, -1), n_neighbors=n_recommendations + 1)
    recommended_songs = []
    for i, idx in enumerate(indices[0][1:], 1):
        song = dataset.iloc[idx]
        recommended_songs.append({
            'id': song['id'],
            'similarity_score': 1 - distances[0][i],
            'year': song['year'],
            'popularity': song['popularity'] * 5
        })
    return recommended_songs

def create_track_html(track, is_input=False):
    try:
        song_name = track['name']
        artist_id = track['artists'][0]['id']
        artist_name = track['artists'][0]['name']
        album_name = track['album']['name']
        release_year = track['album']['release_date'][:4]
        album_cover_url = track['album']['images'][0]['url']
        
        # Fetch artist information to get genres
        artist_info = sp.artist(artist_id)
        genres = ', '.join(artist_info['genres'][:3]) if artist_info['genres'] else 'Not specified'
        
        border_style = "border: 2px solid #1DB954;" if is_input else ""
        
        return f"""
        <div class="song-container" style="{border_style}">
            <div class="song-content">
                <img src="{album_cover_url}" style="width: 120px; height: 120px; object-fit: cover;">
                <div class="song-info">
                    <h3 class="song-title">{song_name}</h3>
                    <p class="song-details">
                        <span class="detail-label">Artist:</span> {artist_name}<br>
                        <span class="detail-label">Album:</span> {album_name}<br>
                        <span class="detail-label">Year:</span> {release_year}<br>
                        <span class="detail-label">Genres:</span> {genres}
                    </p>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="song-container">
            <div class="song-content">
                <div class="song-info">
                    <h3 class="song-title">Unable to load track information</h3>
                    <p class="song-details">
                        This track may no longer be available on Spotify.
                    </p>
                </div>
            </div>
        </div>
        """

st.title("Spotify Song Recommender")

search_query = st.text_input("Search for a song or artist")
if search_query:
    filtered_data = data[data['name'].str.contains(search_query, case=False) | 
                         data['artists'].str.contains(search_query, case=False)]
    search_results = filtered_data['name'] + " by " + filtered_data['artists'].apply(eval).str[0]
    selected_song = st.selectbox("Select a song", options=search_results, format_func=lambda x: x)
else:
    selected_song = None

if selected_song and st.button("Get Recommendations"):
    selected_song_id = filtered_data[filtered_data['name'] + " by " + filtered_data['artists'].apply(eval).str[0] == selected_song]['id'].values[0]
    
    st.markdown("## Selected Song")
    try:
        selected_track = sp.track(selected_song_id)
        st.markdown(create_track_html(selected_track, is_input=True), unsafe_allow_html=True)

        recommendations = get_recommendations(nn_model, data, preprocessed_data, selected_song_id)
        st.markdown("## Recommendations")
        for rec in recommendations:
            try:
                track = sp.track(rec['id'])
                st.markdown(create_track_html(track), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Unable to load recommendation: {e}")
                continue
    except Exception as e:
        st.error(f"Unable to load selected track: {e}")
elif not selected_song:
    st.write("Please select a song to get recommendations.")