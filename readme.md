# Campus IoT Anomaly Detection

## Quick start (dashboard)

**Run the backend first**, then the frontend.

> **Note:** the backend’s data generator can be configured via the
> `DATA_GENERATION_MODE` environment variable.  See below for examples.

1. **Backend** (from repo root):
   ```bash
   cd C-Backend/P1
   uvicorn main:app --reload
   ```
   API runs at http://localhost:8000

   Scripts now default to random mode automatically.  You can still
override it if you prefer:

```bash
DATA_GENERATION_MODE=template ./scripts/macos/run-all-services.sh
# or on Windows powershell
$env:DATA_GENERATION_MODE = "mixed"; .\scripts\run-all-services.ps1
```

The three options are:

| mode     | behaviour |
|----------|-----------|
| `template` | sample from training set with small perturbations |
| `random`   | generate heavily‑perturbed records (this is now the implicit default) |
| `mixed`    | mostly template but occasionally injects an extreme anomaly |

Running the script without setting anything will start in `random` mode
thanks to the updated wrappers, so you don't need to remember to add the
environment variable any more.

2. **Frontend** (in a new terminal):
   ```bash
   cd D-Dashboard/V6
   npm install
   npm run dev
   ```
   Open http://localhost:5174 in your browser.

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
