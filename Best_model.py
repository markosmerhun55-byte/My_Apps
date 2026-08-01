import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle

# Load Dataset
df = pd.read_csv('house.csv')

# Features and Target
X = df.drop(columns=['Price_ETB'])
Y = df['Price_ETB']

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

class LinearPerceptronRegressor:
    def __init__(self, learning_rate=0.001, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0.0
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                y_predicted = np.dot(x_i, self.weights) + self.bias
                error = y[idx] - y_predicted
                self.weights += self.lr * error * x_i
                self.bias += self.lr * error

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2) 
    rmse = np.sqrt(mse)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return mse, rmse, r2

encoders = ["Label", "OneHot"]
scalers = ["NoScaler", "Standard", "MinMax"]
results = []
pipeline_tracker = {}

for enc in encoders:
    if enc == "Label":
        X_enc = X.copy()
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col])
            label_encoders[col] = le
        encoder_obj = label_encoders
    else:
        ct = ColumnTransformer(
            [("onehot", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)],
            remainder="passthrough"
        )
        X_enc = ct.fit_transform(X)
        encoder_obj = ct

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_enc, Y, test_size=0.20, random_state=42
    )

    for scaler_name in scalers:
        scaler_obj = None
        if scaler_name == "NoScaler":
            X_train_scaled = np.array(X_train)
            X_test_scaled = np.array(X_test)
        elif scaler_name == "Standard":
            scaler_obj = StandardScaler()
            X_train_scaled = scaler_obj.fit_transform(X_train)
            X_test_scaled = scaler_obj.transform(X_test)
        elif scaler_name == "MinMax":
            scaler_obj = MinMaxScaler()
            X_train_scaled = scaler_obj.fit_transform(X_train)
            X_test_scaled = scaler_obj.transform(X_test)

        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(alpha=10.0, random_state=42),
            "Lasso": Lasso(alpha=1.0, random_state=42),
            "SVR": SVR(kernel='rbf', C=100000.0, epsilon=0.1),
            "RFR": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "GBM": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
            "Perceptron": LinearPerceptronRegressor(learning_rate=0.001, epochs=100)
        }

        for name, model in models.items():
            try:
                model.fit(X_train_scaled, Y_train)
                predictions = model.predict(X_test_scaled)
                mse, rmse, r2 = calculate_metrics(Y_test, predictions)

                key = f"{enc}_{scaler_name}_{name}"
                pipeline_tracker[key] = {
                    "model": model,
                    "encoder": encoder_obj,
                    "scaler": scaler_obj,
                    "enc_type": enc,
                    "scaler_type": scaler_name
                }

                results.append({
                    "Key": key,
                    "Encoder": enc,
                    "Scaler": scaler_name,
                    "Model": name,
                    "R²": round(r2, 4),
                    "RMSE": round(rmse, 3),
                    "MSE": round(mse, 3)
                })
            except Exception as e:
                print(f"Skipping {name} with {enc} + {scaler_name}: {e}")

results_df = pd.DataFrame(results).sort_values(by="R²", ascending=False).reset_index(drop=True)
print("+++++++++++++++++++++++++++++++++++++++  All 42  Test Dataset +++++++++++++++++++++++++++++++++++")
print(results_df)
best_run = results_df.iloc[0]
print("++++++++++++++++++++++++++++ BEST MODEL  ++++++++++++++++++++++++")
print(best_run)

# Extract and Save Model Artifacts
winning_key = best_run["Key"]
best_pipeline = pipeline_tracker[winning_key]

# Save Metadata for the Streamlit Web App UI
best_pipeline["categorical_cols"] = categorical_cols
best_pipeline["numerical_cols"] = numerical_cols
best_pipeline["feature_names"] = X.columns.tolist()

with open("house_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)
print("\nBest model successfully saved to 'house_model.pkl'")