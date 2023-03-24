import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv('day1prediction.csv')

print(df.columns)

# Split the data into features and target
X = df.iloc[:, :-5] # the last 5 columns are the targets
y = df.iloc[:, -5:]

# Create a mask to filter the data based on the years
train_mask = (df['Year'] >= 2010) & (df['Year'] <= 2016)
test_mask = df['Year'] == 2017

# Split the data into training and testing sets
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

output_columns = ['DJI_Close_Direction', 'NASDAQ_Close_Direction', 'NYSE_Close_Direction', 'RUSSELL_Close_Direction',
                  'SP_Close_Direction']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(X_train_scaled)

lr_models = {}


for col in output_columns:
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train[col])
    y_pred_lr = lr_model.predict(X_test_scaled)

    lr_models[col] = lr_model

    print(f"Logistic Regression Performance for {col}:")
    print("Accuracy:", accuracy_score(y_test[col], y_pred_lr))
    print(classification_report(y_test[col], y_pred_lr))

# Implement the SVM classifier for each output column
svm_results = {}

for col in y_train.columns:
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)  # Adjust the parameters as needed
    svm_model.fit(X_train_scaled, y_train[col])
    y_pred_svm = svm_model.predict(X_test_scaled)

    svm_results[col] = svm_model

    # Evaluate the model
    accuracy = accuracy_score(y_test[col], y_pred_svm)

    print(f"SVM Performance for {col}:")
    print("Accuracy:", accuracy)
    print(classification_report(y_test[col], y_pred_svm))

