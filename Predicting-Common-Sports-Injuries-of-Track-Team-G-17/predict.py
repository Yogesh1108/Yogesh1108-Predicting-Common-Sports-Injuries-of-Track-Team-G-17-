# import numpy as np
# import pandas as pd
#
# # Define constants
# num_samples_per_injury = 3000  # Number of samples per injury type
# selected_injuries = ['Sprain', 'Strain', 'Fracture']
# events = ['Sprint', 'Long-Distance', 'Jumping', 'Throwing']
#
#
# # Function to generate base athlete data
# def generate_base_data(num_samples):
#     ages = np.random.randint(18, 35, size=num_samples)
#     genders = np.random.choice(['Male', 'Female'], size=num_samples)
#     years_of_training = np.random.randint(1, 15, size=num_samples)
#     event_types = np.random.choice(events, size=num_samples)
#     training_hours_per_week = np.random.randint(5, 20, size=num_samples)
#
#     return pd.DataFrame({
#         'Age': ages,
#         'Gender': genders,
#         'Years of Training': years_of_training,
#         'Event Type': event_types,
#         'Training Hours per Week': training_hours_per_week
#     })
#
#
# # Generate balanced data
# balanced_data = pd.DataFrame()
# for injury in selected_injuries:
#     data = generate_base_data(num_samples_per_injury)
#     data['Injury'] = injury
#     balanced_data = pd.concat([balanced_data, data], ignore_index=True)
#
# # Shuffle the dataset
# balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Save to CSV
# balanced_data.to_csv('athlete_injury_data.csv', index=False)


import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load the model, scaler, and label encoders
model = load_model('injury_nn_model.h5')
scaler = joblib.load('scaler_few_classes.pkl')
label_encoder_gender = joblib.load('label_encoder_gender_few_classes.pkl')
label_encoder_event = joblib.load('label_encoder_event_few_classes.pkl')
label_encoder_injury = joblib.load('label_encoder_injury_few_classes.pkl')


# Function to take manual input and make a prediction
def manual_predict():
    # Take manual input
    age = int(input('Enter Age: '))
    gender = input('Enter Gender (Male/Female): ')
    years_of_training = int(input('Enter Years of Training: '))
    event_type = input('Enter Event Type (Sprint/Long-Distance/Jumping/Throwing): ')
    training_hours_per_week = int(input('Enter Training Hours per Week: '))

    # Encode categorical input
    gender_encoded = label_encoder_gender.transform([gender])[0]
    event_type_encoded = label_encoder_event.transform([event_type])[0]

    # Create input array
    input_data = np.array([[age, gender_encoded, years_of_training, event_type_encoded, training_hours_per_week]])

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction_encoded = model.predict(input_data)
    predicted_class = np.argmax(prediction_encoded, axis=1)

    # Decode the prediction
    prediction = label_encoder_injury.inverse_transform(predicted_class)

    print(f'Predicted Injury: {prediction[0]}')


# Call the function to predict
if __name__ == '__main__':
    manual_predict()
