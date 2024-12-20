# Yoga Pose Recommender System

A machine learning-based web application that recommends yoga poses based on the user's mood. The app takes a textual input describing the user's mood, processes it, and suggests an appropriate yoga pose to improve their mental or physical state.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [User Interface](#next-steps)
- [Next Steps](#next-steps)
- [Setup and Installation](#setup-and-installation)

## Project Overview
The Yoga Pose Recommender System leverages natural language processing (NLP) to predict a user's mood based on a given text and then recommends an appropriate yoga pose to alleviate their emotions. This system is designed for users who are looking for yoga poses based on how they are feeling.

### Key Features:
- **Text-based Mood Analysis**: Users can input a sentence describing their mood.
- **Yoga Pose Recommendation**: Based on the mood, the app suggests a yoga pose.
- **Web Interface**: A simple web interface built using Flask, HTML and CSS.

## Directory Structure

![image](https://github.com/user-attachments/assets/de980d9c-4186-4329-b745-39f2e5b59a9c)


## Data Preprocessing
The dataset used for training the model was downloaded from Kaggle: [Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text).

### Preprocessing Steps:
1. **Data Cleaning**: 
   - Removed special characters, URLs, and irrelevant symbols (e.g., Twitter handles like `@username`).
   - Converted all text to lowercase for uniformity.
   - Removed stop words to reduce noise in the data.
   
2. **Text Vectorization**:
   - Used **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text data into numerical features that can be used for model training.

## Model Architecture
After testing some models like Decision Tree, Logistic Regression, XGBoost etc The sentiment classification model is finally built using a **Random Forest Classifier** from the **scikit-learn** library.

### Steps in Model Development:
1. **Data Splitting**: Split the dataset into training and testing sets (80/20 split).
2. **Model Training**: 
   - The Random Forest model was trained using the preprocessed text data (TF-IDF features).
   
The model predicts the sentiment (mood) from the input text, which is then mapped to a predefined list of yoga poses.

## Results
The model performed reasonably well in predicting sentiments, with some challenges in handling imbalanced classes like **anger** and **boredom**.

### Key Performance Metrics:
- **Accuracy**: 40%
- **Precision/Recall/F1-Score** for each class (see full classification report in the app).
- **Confusion Matrix**: Shows which sentiments the model predicted correctly and which ones it confused with others.

### Sentiment to Yoga Pose Mapping:
- **Anger**: Restorative Yoga (Child's Pose)
- **Boredom**: Energizing Yoga (Sun Salutations)
- **Happiness**: Vinyasa Flow
- **Sadness**: Heart-Opening Yoga (Camel Pose)
- **Neutral**: Balanced Yoga (Mountain Pose)
- **Love**: Partner Yoga
- **Worry**: Stress-Relief Yoga (Cat-Cow Pose)
- **Fun**: Creative Yoga (Dancer Pose)
- **Enthusiasm**: Power Yoga
- **Hate**: Relaxation Yoga (Savasana)
- **Relief**: Breathing Exercises (Pranayama)
- **Surprise**: Spontaneous Flow Yoga

## User Interface
   Used FLASK, HTML and CSS to give this project an user interface.The snapshot of the interface is given bellow-
   ![image](https://github.com/user-attachments/assets/7a0bdfd0-ca09-40d2-8988-02342b515918)
   ![image](https://github.com/user-attachments/assets/2bfcc7a8-260f-42cf-8348-4143d9c6ffb7)

## Next Steps
- **Improve Model**: 
  - Experiment with different machine learning models such as **SVM**, **XGBoost**, or **Neural Networks** for improved performance.
- **Data Augmentation**:
  - Collect more data to balance sentiment classes and improve generalization.
- **User Feedback**:
  - Allow users to rate the suggested poses for better recommendations over time.
- **Deploy to Production**:
  - Deploy the app on cloud platforms such as **Heroku** or **AWS** for public access.


## Setup and Installation
Follow the steps below to set up the project on your local machine.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/trisha194/Yoga-Recommender-System-.git
