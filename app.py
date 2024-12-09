from flask import Flask, render_template, request, jsonify
import pickle

# Load the saved Random Forest model
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Define the sentiment-to-yoga mapping
# Define the sentiment-to-yoga mapping with images
sentiment_to_yoga = {
    'anger': {
        'pose': 'Restorative Yoga (Child\'s Pose)',
        'image': 'restorative_pose.jpg'
    },
    'boredom': {
        'pose': 'Energizing Yoga (Sun Salutations)',
        'image': 'sun_salutations.jpg'
    },
    'happiness': {
        'pose': 'Vinyasa Flow',
        'image': 'vinyasa_flow.jpg'
    },
    'sadness': {
        'pose': 'Heart-Opening Yoga (Camel Pose)',
        'image': 'camel_pose.jpg'
    },
    'neutral': {
        'pose': 'Balanced Yoga (Mountain Pose)',
        'image': 'mountain_pose.jpg'
    },
    'love': {
        'pose': 'Partner Yoga',
        'image': 'partner_yoga.jpg'
    },
    'worry': {
        'pose': 'Stress-Relief Yoga (Cat-Cow Pose)',
        'image': 'cat_cow_pose.jpg'
    },
    'fun': {
        'pose': 'Creative Yoga (Dancer Pose)',
        'image': 'dancer_pose.jpg'
    },
    'enthusiasm': {
        'pose': 'Power Yoga',
        'image': 'power_yoga.jpg'
    },
    'hate': {
        'pose': 'Relaxation Yoga (Savasana)',
        'image': 'savasana.jpg'
    },
    'relief': {
        'pose': 'Breathing Exercises (Pranayama)',
        'image': 'pranayama.jpg'
    },
    'surprise': {
        'pose': 'Spontaneous Flow Yoga',
        'image': 'spontaneous_flow.jpg'
    }
}


# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Define the home route
@app.route('/predict', methods=['POST'])
def predict():
    statement = request.form.get('statement')

    if not statement:
        return jsonify({'error': 'No statement provided! Please provide a valid input.'}), 400

    # Vectorize the input statement
    statement_vector = tfidf.transform([statement])

    # Predict the sentiment
    predicted_sentiment = rf_model.predict(statement_vector)[0]

    # Map sentiment to yoga pose and image
    yoga_data = sentiment_to_yoga.get(predicted_sentiment, {
        'pose': 'General Yoga Session',
        'image': 'default_yoga.jpg'
    })

    # Return the result
    return render_template(
        'index.html',
        statement=statement,
        predicted_sentiment=predicted_sentiment,
        assigned_yoga_pose=yoga_data['pose'],
        yoga_image=yoga_data['image']
    )


if __name__ == '__main__':
    app.run(debug=True)
