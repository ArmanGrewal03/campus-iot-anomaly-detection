# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install all service deps into the venv
python -m pip install -r .\01_Data_Ingestion_Service\requirements.txt
python -m pip install -r .\02_Model_Service\requirements.txt
python -m pip install -r .\04_User_Service\requirements.txt
python -m pip install -r .\05_Gateway_Proxy\requirements.txt
python -m pip install -r .\06_Live_Metrics_Service\requirements.txt

# Ensure missing ones seen in errors
python -m pip install python-multipart requests aiokafka pytest pytest-cov

# Run tests with coverage
python -m pytest -q `
  --cov=01_Data_Ingestion_Service `
  --cov=02_Model_Service `
  --cov=04_User_Service `
  --cov=05_Gateway_Proxy `
  --cov=06_Live_Metrics_Service `
  --cov-report=term-missing `
  --cov-report=html

# When done
deactivate


Invoke-Item .\htmlcov\index.html