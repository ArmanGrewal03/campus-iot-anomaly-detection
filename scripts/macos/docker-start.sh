#!/bin/bash

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Starting Docker Services${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$PROJECT_ROOT"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found${NC}"
    exit 1
fi

# Start services
echo -e "${YELLOW}Starting all services...${NC}"
docker-compose up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start services${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Waiting for services to be healthy (10s)...${NC}"
sleep 10

# Check service status
echo ""
echo -e "${CYAN}Service Status:${NC}"
docker-compose ps

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Services are starting!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}Service URLs:${NC}"
echo -e "  - Dashboard:        ${WHITE}http://localhost:5173${NC}"
echo -e "  - API Gateway:       ${WHITE}http://localhost:8003${NC}"
echo -e "  - Data Ingestion:    ${WHITE}http://localhost:8000${NC}"
echo -e "  - Model Service:     ${WHITE}http://localhost:8001${NC}"
echo -e "  - User Service:       ${WHITE}http://localhost:8002${NC}"
echo ""
echo -e "${CYAN}To view logs:${NC}"
echo -e "  ${WHITE}docker-compose logs -f${NC}"
echo ""
echo -e "${CYAN}To stop services:${NC}"
echo -e "  ${WHITE}docker-compose down${NC}"
