#!/bin/bash

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Starting All Services (macOS)${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# set default data generation mode if not already defined
: "${DATA_GENERATION_MODE:=random}"
export DATA_GENERATION_MODE

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Paths to individual scripts
SCRIPT_01="$SCRIPT_DIR/run-01-data-ingestion.sh"
SCRIPT_02="$SCRIPT_DIR/run-02-model-service.sh"
SCRIPT_03="$SCRIPT_DIR/run-03-dashboard.sh"
SCRIPT_04="$SCRIPT_DIR/run-04-user-service.sh"
SCRIPT_05="$SCRIPT_DIR/run-05-gateway.sh"
SCRIPT_06="$SCRIPT_DIR/run-06-live-metrics.sh"

# Check if scripts exist
for SCRIPT in "$SCRIPT_01" "$SCRIPT_02" "$SCRIPT_03" "$SCRIPT_04"; do
    if [ ! -f "$SCRIPT" ]; then
        echo -e "${RED}Error: Script not found: $SCRIPT${NC}"
        exit 1
    fi
done

echo -e "${GREEN}Starting services in separate Terminal windows...${NC}"
echo ""
echo -e "${CYAN}Service URLs:${NC}"
echo -e "  - Data Ingestion API: ${WHITE}http://127.0.0.1:8000${NC}"
echo -e "  - Model API:          ${WHITE}http://127.0.0.1:8001${NC}"
echo -e "  - User Service:       ${WHITE}http://127.0.0.1:8002${NC}"
echo -e "  - API Gateway:        ${WHITE}http://127.0.0.1:8003 (optional)${NC}"
echo -e "  - Live Metrics:       ${WHITE}http://127.0.0.1:8010 (optional)${NC}"
echo -e "  - Dashboard:          ${WHITE}http://127.0.0.1:5173 (will open automatically)${NC}"
echo ""

# Function to run a script in a new Terminal window
run_in_new_window() {
    local script_path=$1
    local title=$2
    echo -e "${YELLOW}Starting $title...${NC}"
    osascript -e "tell application \"Terminal\" to do script \"$script_path\"" > /dev/null
    sleep 2
}

run_in_new_window "$SCRIPT_01" "Data Ingestion Service"
run_in_new_window "$SCRIPT_02" "Model Service"
run_in_new_window "$SCRIPT_04" "User Service"
run_in_new_window "$SCRIPT_03" "Dashboard"

if [ -f "$SCRIPT_05" ]; then
    run_in_new_window "$SCRIPT_05" "API Gateway"
fi

if [ -f "$SCRIPT_06" ]; then
    run_in_new_window "$SCRIPT_06" "Live Metrics Service"
fi

echo ""
echo -e "${GREEN}All services are starting...${NC}"
echo -e "${YELLOW}Waiting for services to initialize (10s)...${NC}"
sleep 10

# Open dashboard in browser
DASHBOARD_URL="http://localhost:5173/Home"
echo ""
echo -e "${CYAN}Opening dashboard in browser: $DASHBOARD_URL${NC}"
open "$DASHBOARD_URL" 2>/dev/null || echo -e "${YELLOW}Could not open browser automatically. Please visit $DASHBOARD_URL${NC}"

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}Services are running!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${YELLOW}To stop services, you can close the terminal windows or use the cleanup script.${NC}"

# Create a quick cleanup script as well
CLEANUP_SCRIPT="$SCRIPT_DIR/stop-all-services.sh"
cat > "$CLEANUP_SCRIPT" <<EOF
#!/bin/bash
echo "Stopping services on known ports..."
for port in 8000 8001 8002 8003 8010 5173; do
    pid=\$(lsof -t -i:\$port)
    if [ ! -z "\$pid" ]; then
        echo "Killing process on port \$port (PID: \$pid)"
        kill -9 \$pid 2>/dev/null
    fi
done
echo "Done."
EOF
chmod +x "$CLEANUP_SCRIPT"

echo -e "A cleanup script has been created at: ${WHITE}$CLEANUP_SCRIPT${NC}"
echo -e "Run it to stop all service processes."
echo ""
