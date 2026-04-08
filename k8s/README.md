Kubernetes deployment for Campus IoT Anomaly Detection

Prerequisites
- Build and push images to a registry accessible by your cluster, or load into your cluster runtime.
  Example (Docker Hub):
  - docker build -t <registry>/data-ingestion:latest 01_Data_Ingestion_Service
  - docker build -t <registry>/model-service:latest 02_Model_Service
  - docker build -t <registry>/user-service:latest 04_User_Service
  - docker build -t <registry>/api-gateway:latest 05_Gateway_Proxy
  - docker build -t <registry>/live-metrics:latest 06_Live_Metrics_Service
  - docker build -t <registry>/dashboard:latest 03_Dashboard
  - docker push … (for each)
  Then update image: fields in deploy.yaml to use your <registry>/… names.

Apply
1) kubectl apply -f k8s/deploy.yaml
2) Verify: kubectl -n campus-iot get pods,svc,ingress
3) If using nginx ingress locally, add hosts entry:
   127.0.0.1 campus.local
4) Access:
   - Dashboard: http(s)://campus.local/
   - Gateway:   http(s)://campus.local/gateway/health

Notes
- Storage: SQLite files are ephemeral (emptyDir). For persistence, replace the emptyDir volumes with PVCs and mount dedicated paths for each service’s DB files.
- User Service disables Kafka/WebSocket by default in this manifest (MESSAGE_QUEUE_ENABLED=false, WEBSOCKET_ENABLED=false). Enable and configure if needed.
- Gateway environment points to internal ClusterIP services; adjust if you change service names/ports.

