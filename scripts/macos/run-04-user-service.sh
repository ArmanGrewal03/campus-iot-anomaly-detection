#!/bin/bash

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Starting 04 User Service${NC}"
echo -e "${CYAN}========================================${NC}"

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
SERVICE_DIR="$PROJECT_ROOT/04_User_Service"

# Check if service directory exists
if [ ! -d "$SERVICE_DIR" ]; then
    echo -e "${RED}Error: Service directory not found at $SERVICE_DIR${NC}"
    exit 1
fi

# Change to service directory
cd "$SERVICE_DIR"

# Check if virtual environment exists
VENV_PATH="$SERVICE_DIR/venv"
if [ ! -d "$VENV_PATH" ] || [ ! -f "$VENV_PATH/bin/python" ]; then
    if [ -d "$VENV_PATH" ]; then
        echo -e "${YELLOW}Virtual environment appears corrupted. Recreating...${NC}"
        rm -rf "$VENV_PATH"
    else
        echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    fi
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Check if requirements are installed
echo -e "${GREEN}Checking dependencies...${NC}"
REQUIREMENTS_FILE="$SERVICE_DIR/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}Installing/updating requirements...${NC}"
    pip install -q -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Warning: Some dependencies may not have installed correctly${NC}"
    fi
fi

# Start the FastAPI service
echo -e "${GREEN}Starting FastAPI User Service on http://127.0.0.1:8002${NC}"
echo -e "${GREEN}WebSocket endpoint: ws://127.0.0.1:8002/ws/data-stream${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

python3 -m uvicorn user_service:app --host 127.0.0.1 --port 8002 --reload
