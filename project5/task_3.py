import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def generate_sliding_window_dataset(data, window_size):
    X = []
    y = []

    for i in range(window_size, len(data)):
        X.append(data.iloc[i - window_size:i, :-5].values.flatten())  # Flatten the array
        y.append(data.iloc[i, -5:].values)

    return pd.DataFrame(X), pd.DataFrame(y)


# Load the dataset
data = pd.read_csv('day1prediction.csv')


# Set the window size to 2 days
window_size = 1

# Generate the dataset
X_window, y_window = generate_sliding_window_dataset(data, window_size)

# Create a mask to filter the data based on the years
train_mask = (data['Year'] >= 2010) & (data['Year'] <= 2016)
test_mask = data['Year'] == 2017

# Split the data into training and testing sets
X_train = X_window[train_mask]
y_train = y_window[train_mask]
X_test = X_window[test_mask]
y_test = y_window[test_mask]

# Standardize the input data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA to the input data
pca = PCA(n_components=0.95)  # Preserve 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

output_columns = ['DJI_Close_Direction', 'NASDAQ_Close_Direction', 'NYSE_Close_Direction', 'RUSSELL_Close_Direction',
                  'SP_Close_Direction']

# Train an MLPClassifier for each output column
mlp_results = {}

for col in y_train.columns:
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), activation='relu', solver='adam', random_state=42)
    mlp_model.fit(X_train_scaled, y_train[col])
    y_pred_mlp = mlp_model.predict(X_test_scaled)

    mlp_results[col] = mlp_model

    # Evaluate the model
    accuracy = accuracy_score(y_test[col], y_pred_mlp)

    print(f"MLPClassifier Performance for {output_columns[col]}:")
    print("Accuracy:", accuracy)
    print(classification_report(y_test[col], y_pred_mlp))
