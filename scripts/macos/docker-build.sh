#!/bin/bash

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Building Docker Images${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

echo -e "${YELLOW}Building Data Ingestion Service...${NC}"
docker build -t campus-iot-data-ingestion ./01_Data_Ingestion_Service
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Data Ingestion Service${NC}"
    exit 1
fi

echo -e "${YELLOW}Building Model Service...${NC}"
docker build -t campus-iot-model-service ./02_Model_Service
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Model Service${NC}"
    exit 1
fi

echo -e "${YELLOW}Building User Service...${NC}"
docker build -t campus-iot-user-service ./04_User_Service
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build User Service${NC}"
    exit 1
fi

echo -e "${YELLOW}Building Gateway...${NC}"
docker build -t campus-iot-gateway ./05_Gateway_Proxy
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Gateway${NC}"
    exit 1
fi

echo -e "${YELLOW}Building Dashboard...${NC}"
docker build -t campus-iot-dashboard ./03_Dashboard
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Dashboard${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All images built successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}To start all services, run:${NC}"
echo -e "  ${WHITE}docker-compose up -d${NC}"
echo ""
echo -e "${CYAN}To view logs:${NC}"
echo -e "  ${WHITE}docker-compose logs -f${NC}"
