#!/bin/bash
echo "Stopping services on known ports..."
for port in 8000 8001 8002 8003 8010 5173; do
    pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid 2>/dev/null
    fi
done
echo "Done."
