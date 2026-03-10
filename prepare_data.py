import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

os.makedirs("data", exist_ok=True)

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 1. Save main dataset
df.to_csv("data/iris.csv", index=False)

# 2. Save training reference (Phase 4 monitoring)
df.to_csv("data/train_reference.csv", index=False)

# 3. Save "serving" data (simulated live data with slight drift)
# We'll add some noise to create a drift scenario for the report
serving_df = df.copy()
serving_df["petal length (cm)"] += np.random.normal(1.5, 0.5, size=len(df)) 
serving_df.to_csv("data/serving_data.csv", index=False)

print("✅ Data files created in data/ folder:")
print("   - data/iris.csv")
print("   - data/train_reference.csv")
print("   - data/serving_data.csv")
