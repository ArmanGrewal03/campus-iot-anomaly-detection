# PlantUML Diagrams for Campus IoT Anomaly Detection

This directory contains PlantUML diagrams documenting the system architecture and flows.

## Diagrams

1. **01-system-architecture.puml** - High-level system architecture showing all components, services, and their relationships
2. **02-training-sequence.puml** - Sequence diagram for model training flow
3. **03-prediction-sequence.puml** - Sequence diagram for prediction flow
4. **04-database-schema.puml** - Database schema diagram showing all tables and relationships
5. **05-api-endpoint-flow.puml** - Flow diagram showing all API endpoints and their connections
6. **06-websocket-flow.puml** - WebSocket data stream flow diagram
7. **07-data-ingestion-flow.puml** - Data ingestion flow (CSV upload, validation, JSON insert)

## How to View/Generate

### Option 1: Online Viewer
1. Go to http://www.plantuml.com/plantuml/uml/
2. Copy and paste the contents of any `.puml` file
3. View the rendered diagram

### Option 2: VS Code Extension
1. Install "PlantUML" extension in VS Code
2. Open any `.puml` file
3. Press `Alt+D` or use the command palette to preview

### Option 3: Command Line
```bash
# Install PlantUML (requires Java)
# Download from: https://plantuml.com/download

# Generate PNG
java -jar plantuml.jar diagrams/01-system-architecture.puml

# Generate SVG
java -jar plantuml.jar -tsvg diagrams/01-system-architecture.puml
```

### Option 4: GitHub/GitLab
GitHub and GitLab automatically render PlantUML files when viewed in the repository.

## Diagram Descriptions

### System Architecture
Shows the complete system with:
- Client layer (Flask Dashboard, External Clients)
- API Services (Backend API on port 8000, Model API on port 8001)
- Data Storage (SQLite databases)
- Machine Learning components
- Data sources

### Training Sequence
Shows the complete flow when training the model:
1. Client sends POST /train
2. Model API checks backend health
3. Model API fetches training data from Backend API
4. Model is trained and saved
5. Response returned to client

### Prediction Sequence
Shows the flow for making predictions:
1. Client sends POST /predict with data
2. Model API loads model and metadata
3. Features are prepared and reordered
4. Predictions are made
5. Results returned to client

### Database Schema
Shows all database tables:
- `csv_data` - Stores uploaded CSV data
- `inserted_data` - Stores JSON data from POST /insert
- `websocket_data` - Stores real-time generated data

### API Endpoint Flow
Shows all endpoints and their relationships to databases and models.

### WebSocket Flow
Shows the real-time data generation and streaming process.

### Data Ingestion Flow
Shows the complete flow for:
- CSV file upload
- Data validation (training/testing split)
- JSON data insertion
