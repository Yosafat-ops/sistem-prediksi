# create_dummy_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Buat model dummy
model = RandomForestClassifier()
X_dummy = np.random.rand(10, 5)  # 10 samples, 5 features
y_dummy = np.random.randint(1, 6, 10)  # Target 1-5
model.fit(X_dummy, y_dummy)

# Simpan model
joblib.dump(model, "models/quality_model.pkl")
print("Model dummy telah dibuat!")