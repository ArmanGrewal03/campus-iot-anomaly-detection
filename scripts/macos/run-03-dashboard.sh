#!/bin/bash

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Starting 03 Dashboard (React/Vite)${NC}"
echo -e "${CYAN}========================================${NC}"

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
SERVICE_DIR="$PROJECT_ROOT/03_Dashboard"

# Check if service directory exists
if [ ! -d "$SERVICE_DIR" ]; then
    echo -e "${RED}Error: Service directory not found at $SERVICE_DIR${NC}"
    exit 1
fi

# Change to service directory
cd "$SERVICE_DIR"

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed or not in PATH${NC}"
    echo ""
    echo -e "${YELLOW}Please install Node.js and npm to continue.${NC}"
    echo -e "You can use Homebrew: ${CYAN}brew install node${NC}"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Node modules not found. Installing dependencies...${NC}"
    echo -e "${YELLOW}This may take a few minutes...${NC}"
    npm install
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install npm dependencies${NC}"
        exit 1
    fi
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
fi

# Start the React development server
echo -e "${GREEN}Starting React development server...${NC}"
echo -e "${YELLOW}The dashboard will open automatically in your browser${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

npm run dev
