"""
*** NOT FULLY IMPLEMENTED - PLACEHOLDER ONLY ***
Placeholder for CNN model type (CNN).

This file exists so that the Model Service can advertise a CNN model_type
via GET /model-types (it just scans the Model_types directory).

To fully enable CNN:
- Choose a deep learning framework (PyTorch or TensorFlow/Keras).
- Design a 1D CNN (or similar) for tabular / time-series style data.
- In model_api.py:
  - Implement a /train branch for model_type == "CNN" that trains the network
    and saves weights/checkpoints to the models directory.
  - Extend evaluate_model(...) and /predict to load the CNN and produce
    safe/unsafe labels plus risk scores.

Until that wiring is done, selecting model_type="CNN" will cause /train
to return "Unsupported model type".
"""

