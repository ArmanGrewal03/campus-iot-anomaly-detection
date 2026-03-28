"""
*** NOT FULLY IMPLEMENTED - PLACEHOLDER ONLY ***
Placeholder for LightGBM model type (LIGHTGBM).

This file exists so that the Model Service can advertise a LIGHTGBM model_type
via GET /model-types (it just scans the Model_types directory).

Requirements to fully enable:
- Install lightgbm in the Model Service environment.
- In model_api.py:
  - Import lightgbm.LGBMClassifier.
  - Add a /train branch for model_type == "LIGHTGBM" that trains LGBMClassifier.
  - Extend evaluate_model(...) and /predict to handle LIGHTGBM similar to RFv1.

Until that wiring is done, selecting model_type="LIGHTGBM" will cause /train
to return "Unsupported model type".
"""

