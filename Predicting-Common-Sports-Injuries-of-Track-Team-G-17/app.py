import streamlit as st
import joblib
import numpy as np
from EDA import ExploratoryDataAnalysis

st.set_page_config(page_title='Injury Prediction App', page_icon=':runner:')

from sklearn.preprocessing import LabelEncoder
st.title("COMMON SPORTS INJURIES OF TRACK AND FIELD ATHLETES USING MACHINE LEARNING")
st.image("coverpage.png")
eda = ExploratoryDataAnalysis("athlete_injury_data.csv")
#
# Load the model, scaler, and label encoders
kmeans = joblib.load('injury_kmeans_model.pkl')
scaler = joblib.load('scaler_few_classes.pkl')
label_encoder_gender = joblib.load('label_encoder_gender_few_classes.pkl')
label_encoder_event = joblib.load('label_encoder_event_few_classes.pkl')
label_encoder_injury = joblib.load('label_encoder_injury_few_classes.pkl')

# Map cluster labels to injury types based on the majority class in each cluster
cluster_to_injury = {0: 'Sprain', 1: 'Strain', 2: 'Fracture'}


# Function to make a prediction
def manual_predict(input_tuple):
    age, gender, years_of_training, event_type, training_hours_per_week = input_tuple
    gender_encoded = label_encoder_gender.transform([gender])[0]
    event_type_encoded = label_encoder_event.transform([event_type])[0]
    input_data = np.array([[age, gender_encoded, years_of_training, event_type_encoded, training_hours_per_week]])
    input_data = scaler.transform(input_data)
    cluster = kmeans.predict(input_data)[0]
    predicted_injury = cluster_to_injury.get(cluster, 'Unknown')
    return predicted_injury, cluster


# Login function
def login(username, password):
    if username == 'admin' and password == 'password':  # Simple authentication
        st.session_state['logged_in'] = True
        return True
    else:
        st.session_state['logged_in'] = False
        return False


# Streamlit app


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.subheader('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if login(username, password):
            st.success('Login successful!')
        else:
            st.error('Invalid username or password')
else:
    st.subheader('Predict an Injury')
    eda.run()
    age = st.number_input('Age', min_value=0, max_value=100, value=25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    years_of_training = st.number_input('Years of Training', min_value=0, max_value=50, value=10)
    event_type = st.selectbox('Event Type', ['Sprint', 'Long-Distance', 'Jumping', 'Throwing'])
    training_hours_per_week = st.number_input('Training Hours per Week', min_value=0, max_value=168, value=15)

    input_tuple = (age, gender, years_of_training, event_type, training_hours_per_week)
    if st.button('Predict'):
        predicted_injury, cluster = manual_predict(input_tuple)
        st.success(f'Predicted Injury: {predicted_injury}')
        st.markdown(f'<h2 style="color:green;">Cluster: {cluster}</h2>', unsafe_allow_html=True)

# Website description
st.markdown("""
### Welcome to the Track and Field Athlete Injury Prediction Application!

In the high-intensity world of track and field, athletes are constantly pushing their bodies to the limit. This relentless pursuit of excellence, while commendable, often leads to injuries that can sideline even the most dedicated competitors. Understanding and predicting these injuries can be a game-changer for athletes, coaches, and medical professionals.

This application leverages the power of machine learning, specifically K-Means clustering, to predict common injuries among track and field athletes. By analyzing key factors such as age, gender, years of training, event type, and training hours per week, our model can provide insights into the likelihood of specific injuries.

#### How It Works
1. **Input Data**: Enter the athlete's details including age, gender, years of training, event type, and weekly training hours.
2. **Prediction**: The application processes this data using a pre-trained K-Means clustering model.
3. **Result**: The predicted injury type and the corresponding cluster are displayed, offering valuable information to help mitigate injury risks.

#### Benefits
- **Preventive Measures**: By identifying potential injury risks, preventive measures can be implemented to protect athletes.
- **Training Adjustments**: Coaches can tailor training programs to address and reduce the likelihood of injuries.
- **Enhanced Recovery**: Early predictions of injury types can lead to quicker and more effective treatment plans.

This application is designed with user-friendliness in mind. It features a straightforward login system to ensure secure access. Once logged in, users can easily input data and receive predictions almost instantly.

We are committed to helping athletes achieve their best while staying safe and healthy. Explore the features of our application and see how it can make a difference in your training and injury management strategies.

### Disclaimer
This tool is intended for educational purposes and should not replace professional medical advice. Always consult with a healthcare provider for any injury-related concerns.
""")

# HTML footer
# HTML Footer
st.markdown("""
    <footer style="position: fixed; bottom: 0; width: 100%; background-color: #f1f1f1; text-align: center; padding: 10px;">
    <p>&copy; 2024 Injury Prediction System. All rights reserved.</p>
    </footer>
    """, unsafe_allow_html=True)

