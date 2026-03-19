"""
*** NOT FULLY IMPLEMENTED - PLACEHOLDER ONLY ***
Placeholder for KNN model type (KNN).

This file exists so that the Model Service can advertise a KNN model_type
via GET /model-types (it just scans the Model_types directory).

NOTE:
- Training and inference logic for KNN is NOT yet implemented in model_api.py.
- If you select model_type="KNN" in the UI or API, /train will currently
  return "Unsupported model type".

To fully enable KNN:
- Add a new branch in model_api.py /train for model_type == "KNN"
  that trains a KNeighborsClassifier.
- Extend evaluate_model(...) and /predict to handle the KNN case.
"""

