import pickle
import webbrowser
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from googleapiclient.discovery import build
import random
import cv2
import numpy as np
from tensorflow import keras
from tf_keras.models import model_from_json
from ultralytics import YOLO
from config import CLIENT_ID, CLIENT_SECRET, YOUTUBE_API_KEY


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize the YOLO drowsiness model
drowsiness_model = YOLO('best.pt')

# Define stress-relief keywords
stress_relief_keywords = [
    "relaxing","calm","peaceful", "meditative", "serene", "soothing", "tranquil", "gentle", "mindfulness", "zen",
    "chill", "ambient", "soft", "harmonious", "comforting", "dreamy", "stress relief", "nature sounds", "yoga", "meditation"
]

def get_youtube_url(song_name, artist_name):
    search_query = f"{song_name} {artist_name} official audio"
    search_response = youtube.search().list(q=search_query, type='video', part='id', maxResults=1).execute()
    if 'items' in search_response and search_response['items']:
        video_id = search_response['items'][0]['id']['videoId']
        return f"https://www.youtube.com/watch?v={video_id}"
    return None

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"{song_name} {artist_name} official audio"
    search_response = youtube.search().list(q=search_query, type='video', part='snippet', maxResults=1).execute()
    if 'items' in search_response and search_response['items']:
        video = search_response['items'][0]
        if 'thumbnails' in video['snippet']:
            return video['snippet']['thumbnails']['medium']['url']
    return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(keyword):
    music_list = []
    search_query = f"{keyword} music"
    results = sp.search(q=search_query, type="track", limit=10)
    music_list.extend([(track['name'], track['artists'][0]['name']) for track in results['tracks']['items']])
    
    recommended_music_names, recommended_music_urls, recommended_music_posters = [], [], []
    for song_name, artist_name in music_list:
        recommended_music_names.append(song_name)
        recommended_music_urls.append(get_youtube_url(song_name, artist_name))
        recommended_music_posters.append(get_song_album_cover_url(song_name, artist_name))
    
    return recommended_music_names, recommended_music_urls, recommended_music_posters

# Function to predict physical stress
def phyPredict(string):
    strarr = string.split(",")
    arr = np.array([float(ele) for ele in strarr])
    global physicalAtrributesModel
    result = physicalAtrributesModel.predict(arr.reshape(1,-1)).flatten()
    return result[0]

    
def facePredict(video) :
    if (video=="Stressed") :
      cap = cv2.VideoCapture("Inputs/stressed.mp4")
    elif (video=="Not Stressed"):
      cap = cv2.VideoCapture("Inputs/non_stressed.mp4")

    flag = True

    while (flag):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            emotion_prediction = emotion_model.predict(cropped_img)
            maxIndex = int(np.argmax(emotion_prediction))

            if (maxIndex in [0,1,2]):
                cap.release()
                cv2.destroyAllWindows()
                return 1
                flag = False
            else :
                cap.release()
                cv2.destroyAllWindows()
                return 0
                flag = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
def drowsinessDetection(video, drowsy_threshold, non_drowsy_threshold):
    if (video=="Drowsy") :
      cap = cv2.VideoCapture("Inputs/drowsy1.mp4")
    elif (video=="Not Drowsy"):
      cap = cv2.VideoCapture("Inputs/non_drowsy.mp4")
    
    drowsy_frame_count = 0  
    non_drowsy_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            cap.release()
            return False  

        
        results = drowsiness_model.predict(source=frame, conf=0.25, save=False)

        drowsy_detected = False
        for result in results:
            for detection in result.boxes:
                label = drowsiness_model.names[int(detection.cls.cpu().numpy())]
                if label == "drowsy":
                    drowsy_detected = True
                    break

        # Increment the drowsy frame count if drowsiness is detected
        if drowsy_detected:
            non_drowsy_frame_count = 0
            drowsy_frame_count += 1
        else:
            non_drowsy_frame_count += 1
            drowsy_frame_count = 0  

        if drowsy_frame_count >= drowsy_threshold:
            cap.release()
            return True 

        if non_drowsy_frame_count >= non_drowsy_threshold:
            cap.release()
            return False 

# Streamlit app
st.header('Driver Stress & Drowsiness Management')

drowsyInput = st.radio("Drowsy Detection Video Input", ["Drowsy", "Not Drowsy"])
faceInput = st.radio("Emotion Detection Video Input", ["Stressed", "Not Stressed"])
phyInput = st.text_input('Physical Attributes Input')

json_file = open('Models/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("Models/emotion_model.h5")

physicalAtrributesModel = pickle.load(open('Models/phyattrmodel2.pkl', 'rb'))

# Define a dropdown for selecting stress-relief keywords
selected_keyword = None  # Initialize selected_keyword as None

# Submit button logic
if st.button('Submit'):
    if drowsinessDetection(drowsyInput, 10, 10):
        st.text("Driver is drowsy! ALERT!")
        st.audio("Inputs/alert.mp3", autoplay=True)
    else:
        # Check for physical stress
        if phyPredict(phyInput) == 2:  # Physical stress confirmed
            # Check for emotional stress only if physical stress is confirmed
            if facePredict(faceInput) == 1:  # Emotional stress confirmed
                # Display the dropdown for stress-relief keyword selection
                selected_keyword = st.selectbox("Select a stress-relief keyword for song recommendations", stress_relief_keywords)
                
                # Show recommendations based on the selected keyword
                if selected_keyword:
                    recommended_music_names, recommended_music_urls, recommended_music_posters = recommend(selected_keyword)
                    if recommended_music_urls:
                        webbrowser.open_new_tab(recommended_music_urls[0])  # Play the first recommended song
                    col1, col2, col3, col4, col5 = st.columns(5)
                    for i in range(5):
                        with [col1, col2, col3, col4, col5][i]:
                            st.text(recommended_music_names[i])
                            st.image(recommended_music_posters[i])
                            st.success(f"[Play Song]({recommended_music_urls[i]})", icon="🎶")
            else:
                st.text("Driver is Not Emotionally Stressed.")
        else:
            st.text("Driver is Not Physically Stressed.")
