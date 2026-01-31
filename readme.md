# Campus IoT Anomaly Detection

## Quick start (dashboard)

**Run the backend first**, then the frontend.

1. **Backend** (from repo root):
   ```bash
   cd C-Backend/P1
   uvicorn main:app --reload
   ```
   API runs at http://localhost:8000

2. **Frontend** (in a new terminal):
   ```bash
   cd D-Dashboard/V6
   npm install
   npm run dev
   ```
   Open http://localhost:5173 in your browser.

---

## 1. Data Processing
Run the preprocessing script to clean and encode the UNSW-NB15 dataset.

```bash
python3 /Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Scripts/ProcessData.py \
  --input_csv /Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Data/UNSW_NB15_testing-set.csv \
  --out_dir /Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Processed \
  --make_split
```

## 2. Model Training (Random Forest v1)
Train the Random Forest model using the processed data.

```bash
python3 /Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/B-Model/RFv1/train_rf.py \
  --data_dir /Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/A-DataIngestion/Processed \
  --out_dir /Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/B-Model/RFv1/output \
  --threshold 0.55 \
  --threshold_sweep
```

## Model Artifacts
Training artifacts (model, metrics, sample output) are saved to:
`/Users/armangrewal/Desktop/capstone/campus-iot-anomaly-detection/B-Model/RFv1/output`
