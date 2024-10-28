````markdown
# AI-Driven Drowsiness and Stress Detection with Auto Music Recommendation

This project is an **AI-powered drowsiness and stress detection system** designed to enhance driver safety. Using machine learning algorithms, the system analyzes physical and emotional indicators of stress and drowsiness, then automatically recommends and plays soothing music to help the driver relax.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Future Work](#future-work)

## Project Overview

The system leverages **real-time video analysis** and **physiological data** to determine the driver's state:

- **Drowsiness Detection**: Uses YOLO to detect eye closure patterns associated with drowsiness.
- **Stress Detection**: Uses physical attributes and facial expressions to classify the driver’s stress level.
- **Music Recommendation**: If stress is detected, it auto-recommends and plays relaxing music to help the driver.

This end-to-end solution is designed for **long drives and high-stress driving situations** to enhance safety and well-being.

## Key Features

1. **Real-Time Drowsiness Detection**: Alerts the driver if signs of drowsiness are detected.
2. **Stress Detection**: Analyzes physical data like heart rate and temperature, and facial expressions to determine stress levels.
3. **Auto Music Recommendation**: Automatically recommends and plays soothing music if the driver shows signs of stress.
4. **User-Friendly Interface**: Interactive frontend created with Streamlit for easy use.

## Technologies Used

- **YOLO (Drowsiness Detection)**: For detecting drowsiness based on eye closure.
- **LightGBM**: Used for physical stress detection from physiological attributes.
- **Custom CNN**: For emotion recognition based on facial expression analysis.
- **Spotify API & YouTube API**: For personalized song recommendations.
- **Streamlit**: Web interface for real-time interaction with the system.

## Dataset

- **WESAD**: A publicly available dataset for stress detection, utilized to train the LightGBM model on physiological attributes.
- **Custom Emotion Data**: Trained a CNN model on facial expressions for emotion recognition.

## Installation and Setup

### 1. Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Pip
- [Git](https://git-scm.com/)

### 2. Clone the Repository

```bash
git clone https://github.com/VKunjir/AI_Driven-Drowsiness-and-Stress-Detection-with-Auto-Music-Recommendation.git
cd AI_Driven-Drowsiness-and-Stress-Detection-with-Auto-Music-Recommendation
```
````

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

1. Create a `.env` file to store your sensitive keys:
   ```bash
   touch .env
   ```
2. Add your keys in the `.env` file:
   ```plaintext
   SPOTIFY_CLIENT_ID="your_spotify_client_id"
   SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
   YOUTUBE_API_KEY="your_youtube_api_key"
   ```
3. **Optional**: Ensure the `.env` file is in your `.gitignore` to prevent accidental uploads.

### 5. Set up Models

- Download the `YOLO` model file (`best.pt`) and save it in the project's root directory.
- Place your pre-trained models (`emotion_model.json`, `emotion_model.h5`, `phyattrmodel2.pkl`) in the `Models` directory.

### 6. Run the Streamlit App

```bash
streamlit run app.py
```

The app should now be running locally on [http://localhost:8501](http://localhost:8501).

## Usage

1. **Drowsiness Detection**: Select a video sample for drowsiness detection and run the system. If drowsiness is detected, an alert is triggered.
2. **Stress Detection**:
   - Enter physical attributes (e.g., heart rate, temperature).
   - Select an emotion-based video sample for stress detection.
   - If stress is detected, select a music keyword and receive song recommendations.
3. **Music Recommendation**: Choose a stress-relief keyword and click on the recommendations to play a relaxing song.

## Future Work

Potential improvements include:

- **Enhanced Personalization**: More tailored song suggestions based on user preferences and stress levels.
- **IoT Integration**: Direct integration with smart home/car devices for seamless alerts and music playback.
- **Advanced Analytics**: Adding predictive analytics for better stress trend analysis.

---
