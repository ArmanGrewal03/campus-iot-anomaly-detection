COMPLETE
# Health
GET /health
- verify server healthy

# Data Ingestion to database
POST /new
- FastAPI with binary
- SQLite

# view all data
GET /view

# validate
PUT validate
- adds training and testing labels

# Get Training Data
GET /training
- FastAPI with SQLite reads

# Get Testing Data
GET /testing
- FastAPI with Sqlite reads


# Clear data
POST /clear
- clear data in database


INCOMPLETE




# Get My Data
get /MyData
- FastAPI to get my data

# Insert row
- actual data (different table)





   cd C-Backend/P1
   .\venv\Scripts\Activate.ps1

    uvicorn main:app --reload