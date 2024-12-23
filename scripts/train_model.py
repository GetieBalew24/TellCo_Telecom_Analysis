import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start MLFlow run
with mlflow.start_run():
    # Log parameters
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train model
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    end_time = time.time()
    
    # Log metrics
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("training_time", end_time - start_time)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts (optional)
    with open("model_summary.txt", "w") as f:
        f.write(f"Model Summary:\nAccuracy: {acc}\n")
    mlflow.log_artifact("model_summary.txt")
