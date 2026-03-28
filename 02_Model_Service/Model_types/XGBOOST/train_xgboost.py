"""
*** NOT FULLY IMPLEMENTED - PLACEHOLDER ONLY ***
Placeholder for XGBoost model type (XGBOOST).

This file exists so that the Model Service can advertise an XGBOOST model_type
via GET /model-types (it just scans the Model_types directory).

Requirements to fully enable:
- Install xgboost in the Model Service environment.
- In model_api.py:
  - Import xgboost.XGBClassifier.
  - Add a /train branch for model_type == "XGBOOST" that trains XGBClassifier.
  - Extend evaluate_model(...) and /predict to handle XGBOOST similar to RFv1.

Until that wiring is done, selecting model_type="XGBOOST" will cause /train
to return "Unsupported model type".
"""

