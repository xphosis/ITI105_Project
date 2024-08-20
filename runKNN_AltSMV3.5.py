# python -m streamlit run runKNN.py

# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
# import pickle
import time
# from thefuzz import fuzz
from thefuzz import process
from sklearn.neighbors import NearestNeighbors



# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name to get similar song recommendations based on music content similarity")



# Dropdown for selecting User ID
user_id = st.selectbox(
    'Please log in with your User ID',
    options=[1001, 1002, 1003, 1004, 1005]
)



#Load Methods
@st.cache_data()
def get_track_data():

    # track_data =pd.read_pickle('track_data_KMeansNearestNeighbors.pkl')
    track_data = pd.read_csv('track_data_KMeansNearestNeighbors.csv')
    return track_data


def get_features_info():
    INPUT_FEATURES = ['id','song', 'artists', 'genres', 'year', 'explicit', 'popularity', 'danceability',
                'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness',  'valence', 'segment']
    
    # TRAIN_FEATURES = ['explicit', 'popularity', 'danceability',
    #         'energy', 'loudness', 'speechiness', 'acousticness',
    #         'instrumentalness', 'valence']
    
    # INPUT_FEATURES = ['id','song', 'artists', 'genres', 'year', 'duration_ms', 'explicit', 'popularity', 'danceability',
    #             'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    #             'instrumentalness', 'liveness', 'valence', 'tempo', 'segment']
    
    TRAIN_FEATURES = ['duration_ms', 'explicit', 'popularity', 'danceability',
            'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo']

    TRACK_SUMMARY = ['id','song', 'artists', 'genres', 'year', 'segment']
    return INPUT_FEATURES,TRAIN_FEATURES, TRACK_SUMMARY


def get_track_info(track_data, selected_song_index):
    
    track_info = track_data[['id','song','artists','genres','year']].iloc[selected_song_index]
    
    return track_info


def get_neighbors_by_segment(selected_songs_by_segment, selected_song_input, no_neighbors=10):
    
        
    nneigh = NearestNeighbors(n_jobs=-1, leaf_size=30, metric='manhattan', algorithm = 'kd_tree', p=1, radius=1.6)
    nneigh.fit(X=selected_songs_by_segment)
    distances, indices_kneighbors =  nneigh.kneighbors(X=selected_song_input,n_neighbors=no_neighbors,return_distance=True)

    indices_kneighbors = indices_kneighbors.tolist()[0]
    indices = []
    for i in indices_kneighbors:
        indices.append(selected_songs_by_segment.index[i])
    
    return distances.tolist()[0], indices


def song_recommender(track_data, user_input=None, no_neighbors=10):


    INPUT_FEATURES, TRAIN_FEATURES, TRACK_SUMMARY = get_features_info()
    select_song_index= process.extractOne(user_input, track_data.song)[2]
    select_song_segment = track_data.segment.iloc[select_song_index]
    # print(f'Song Segment : {select_song_segment} | Song Index : {select_song_index} | Song Released Year : {track_data.year[select_song_index]}')
    # print(f'Song Selected: {track_data.song[select_song_index]} | Song Artist: {track_data.artists[select_song_index]} | Song Genre: {track_data.genres[select_song_index]}')
    # print(f'Searching for recommendations.....')
    selected_song_input = track_data[TRAIN_FEATURES].iloc[select_song_index].values.reshape(1,-1)
    # print(selected_song_input)
    selected_songs_by_segment = track_data[TRAIN_FEATURES].loc[track_data.segment == select_song_segment]
    # print(selected_songs_by_segment)
    distances, indices = get_neighbors_by_segment(selected_songs_by_segment,selected_song_input,no_neighbors)
    
    return distances, indices


def get_recommended_songs(indices, track_data):
    recommend_songs_list = {'song_name': [],
                        'song_artists': [],
                        'song_genres': [],
                        'song_year':[],
                        'song_id': []}
    for idx in indices:
        track_info = get_track_info(track_data=track_data, selected_song_index=idx)
        recommend_songs_list['song_id'].append(track_info['id'])
        recommend_songs_list['song_name'].append(track_info['song'])
        recommend_songs_list['song_artists'].append(track_info['artists'])
        recommend_songs_list['song_genres'].append(track_info['genres'])
        recommend_songs_list['song_year'].append(track_info['year'])

    recommend_songs = pd.DataFrame(recommend_songs_list)

    return recommend_songs


# Load your preprocessed dataset
df = get_track_data()  # Preprocessed music data with numerical features

# Initialize session state for playlist
if 'playlist' not in st.session_state:
    st.session_state.playlist = pd.DataFrame(columns=['Song', 'Artist', 'Genre', 'Year'])

# Input field for song name
song_name_input = st.text_input("Enter a song that you like:")

if song_name_input:
    distances, indices = song_recommender(track_data=df, 
                                user_input=song_name_input,
                                # no_neighbors=st.sesion_state.no_neighbors
                                )
    table_df = get_recommended_songs(indices, df)
    st.write(f'Song Selected by User :  {table_df.iloc[0].song_name} | Artist/s : {table_df.iloc[0].song_artists} ') 
    st.write("Here are some recommended songs that you may like:")
    
    # st.write(f'Top {st.session_state.no_neighbors} Recommended Songs by Segment') 
    # st.dataframe(recommend_songs.sample(n=st.session_state.no_neighbors).sort_index(), use_container_width=True, hide_index=True)
    # Reset index and drop the old index column
    table_df_reset = table_df.head(10).reset_index(drop=True)
    
    # Display the DataFrame without the index column
    st.dataframe(table_df_reset, use_container_width=True, hide_index=True)
    
    filtered_df = table_df.iloc[1:11].reset_index(drop=True)
    
    # Display the filtered table with checkboxes for selection
    st.write("You may select any recommended songs below and click on the 'Add to Playlist' button to create your personal playlist")

    # Display the filtered DataFrame with checkboxes
    selected_indices = []

    # for i in range(1, 11, step=1):

    for idx, row in filtered_df.iterrows():
        song_name = row.get('song_name', 'Unknown Song')
        artist_name = row.get('song_artists', 'Unknown Artist')
        if st.checkbox(f"{song_name} by {artist_name}", key=idx):
            selected_indices.append(idx)

    # Filter selected songs
    selected_songs = filtered_df.loc[selected_indices].drop(columns='song_id')

# If the user clicks the "Add to Playlist" button, show the selected songs
if st.button('Add to Playlist'):
    if not selected_songs.empty:        
        # Add the original song to the playlist DataFrame
        original_song = pd.DataFrame([{
            'Song': df['song'][idx],
            'Artist': df['artists'][idx],
            'Genre': df['genres'][idx],
            'Year': df['year'][idx],
        }])
        
        # Add selected songs
        selected_songs = selected_songs.rename(columns={'song_name': 'Song', 'song_artists': 'Artist', 'song_genres': 'Genre', 'song_year': 'Year' })
        selected_songs['Original Song'] = False
        
        # Combine the original song and selected songs
        final_playlist = pd.concat([original_song, selected_songs], ignore_index=True)

        # Display the playlist
        st.write("Your Playlist:")
        st.dataframe(final_playlist, use_container_width=True, hide_index=True)
        
        # Save the updated playlist to a CSV file
        if user_id:
            filename = f'Playlist_{user_id}.csv'
            final_playlist.to_csv(filename, index=False)
            st.write(f"Playlist saved as {filename}")
        else:
            st.write("User ID is not set. Cannot save playlist.")
    else:
        st.write("No songs selected, please try again.")


# songs = ["Song 1","Song 2","Song3"]
# Create a radio button for each song
# selected_songs = st.radio("Select a song to add to your playlist:", options=songs)

# Create a slider
# rating = st.slider("Please rate the recommended song (5 being Highest", min_value=1, max_value=5, value=1)

# Display the selected value
# st.write("You have given a rating of ", rating)


# # Adding a sidebar
# st.sidebar.title("Sidebar")
# option = st.sidebar.selectbox(
#     'Please enter your User ID',
#     list(range(1001, 1006)))
# st.image('pic.jpg')
# Age=st.sidebar.radio('Please enter your Age Group',options=['Under 20','20+','30+','40+','Over 50'])
