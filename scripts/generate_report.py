import mlflow
import pandas as pd

# Get the experiment data
experiment_id = "0"  # Replace with your experiment ID
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=[experiment_id])

# Extract relevant information
data = []
for run in runs:
    data.append({
        "run_id": run.info.run_id,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "artifacts": run.info.artifact_uri,
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("tracking_report.csv", index=False)
print("Report saved as 'tracking_report.csv'")
