import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

@st.cache_data()

def get_track_data():

    track_data =pd.read_pickle('track_data_KMeansNearestNeighbors.pkl')
    return track_data


def get_features_info():
    INPUT_FEATURES = ['id','song', 'artists', 'genres', 'year', 'explicit', 'popularity', 'danceability',
                'energy', 'loudness', 'speechiness', 'acousticness',
                'instrumentalness',  'valence', 'segment']
    
    TRAIN_FEATURES = ['explicit', 'popularity', 'danceability',
            'energy', 'loudness', 'speechiness', 'acousticness',
            'instrumentalness', 'valence']
    
    # INPUT_FEATURES = ['id','song', 'artists', 'genres', 'year', 'duration_ms', 'explicit', 'popularity', 'danceability',
    #             'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    #             'instrumentalness', 'liveness', 'valence', 'tempo', 'segment']
    
    # TRAIN_FEATURES = ['duration_ms', 'explicit', 'popularity', 'danceability',
    #         'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    #         'instrumentalness', 'liveness', 'valence', 'tempo']

    TRACK_SUMMARY = ['id','song', 'artists', 'genres', 'year', 'segment']
    return INPUT_FEATURES,TRAIN_FEATURES, TRACK_SUMMARY


def get_track_info(track_data, selected_song_index):
    
    track_info = track_data[['id','song','artists','genres','year']].iloc[selected_song_index]
    
    return track_info


def get_neighbors_by_segment(selected_songs_by_segment, selected_song_input, no_neighbors=10):
    from sklearn.neighbors import NearestNeighbors
        
    nneigh = NearestNeighbors(n_jobs=-1, leaf_size=30, metric='manhattan', algorithm = 'kd_tree', p=1, radius=1.6)
    nneigh.fit(X=selected_songs_by_segment)
    distances, indices_kneighbors =  nneigh.kneighbors(X=selected_song_input,n_neighbors=no_neighbors,return_distance=True)

    indices_kneighbors = indices_kneighbors.tolist()[0]
    indices = []
    for i in indices_kneighbors:
        indices.append(selected_songs_by_segment.index[i])
    
    return distances.tolist()[0], indices


def song_recommender(track_data, user_input=None, no_neighbors=10):
    from thefuzz import fuzz
    from thefuzz import process

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
    distances, indices = get_neighbors_by_segment(selected_songs_by_segment,selected_song_input,no_neighbors*5)
    
    return distances, indices


def get_recommended_songs(indices):
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


# track_data = get_track_data()
# trained_model = get_trained_model()

with st.spinner('Initializing ...... ITS105 SONG RECOMMENDER APP DEMO by TEAM 17'):
    track_data = get_track_data()
    time.sleep(1)

st.success('ITS105 SONG RECOMMENDER APP DEMO by TEAM 17')

form = st.form(key='song_recommender_form')
user_input = form.text_input('Enter Anything', key = 'user_input')
no_neighbors = form.slider(label='Select No. of Song Recommendations', min_value=10, max_value=50, key='no_neighbors')
submitted = form.form_submit_button(label='Recommend Songs')

form.write('Hit Recommended Songs for some good listening')

if submitted:
    progress_text = 'Analysing in progress.... Please wait.'
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    distances, indices = song_recommender(track_data=track_data, 
                                user_input=st.session_state.user_input,
                                no_neighbors=st.session_state.no_neighbors
                                )
    recommend_songs = get_recommended_songs(indices)
    st.write(f'Top {st.session_state.no_neighbors} Recommended Songs by Segment', recommend_songs.sample(n=st.session_state.no_neighbors).sort_index()) 
     
    time.sleep(1)
    my_bar.empty()

else:
    st.write(f'Please key your song recommendation preferences!')
