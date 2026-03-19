"""
*** NOT FULLY IMPLEMENTED - PLACEHOLDER ONLY ***
Placeholder for K-Means model type (KMEANS).

This file exists so that the Model Service can advertise a KMEANS model_type
via GET /model-types (it just scans the Model_types directory).

NOTE:
- Training and inference logic for K-Means is NOT yet implemented in model_api.py.
- If you select model_type="KMEANS" in the UI or API, /train will currently
  return "Unsupported model type".

To fully enable K-Means:
- Add a new branch in model_api.py /train for model_type == "KMEANS"
  that trains a sklearn.cluster.KMeans model.
- Decide how to convert clusters into safe/unsafe labels or anomaly scores.
"""

