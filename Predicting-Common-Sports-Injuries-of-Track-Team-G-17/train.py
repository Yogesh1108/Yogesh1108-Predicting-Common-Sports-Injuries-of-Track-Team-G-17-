import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib

# Load the data
data = pd.read_csv('athlete_injury_data.csv')

# Encode categorical variables
label_encoder_gender = LabelEncoder()
label_encoder_event = LabelEncoder()
label_encoder_injury = LabelEncoder()

data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
data['Event Type'] = label_encoder_event.fit_transform(data['Event Type'])

# Features and target
X = data.drop('Injury', axis=1)
y = label_encoder_injury.fit_transform(data['Injury'])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Save the model, scaler, and label encoders
joblib.dump(kmeans, 'injury_kmeans_model.pkl')
joblib.dump(scaler, 'scaler_few_classes.pkl')
joblib.dump(label_encoder_gender, 'label_encoder_gender_few_classes.pkl')
joblib.dump(label_encoder_event, 'label_encoder_event_few_classes.pkl')
joblib.dump(label_encoder_injury, 'label_encoder_injury_few_classes.pkl')

# Print the cluster centers
print('Cluster Centers:', kmeans.cluster_centers_)
