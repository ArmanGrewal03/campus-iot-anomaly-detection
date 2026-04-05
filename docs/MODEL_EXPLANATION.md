# Model Prediction Explanation

## What the Models Predict

The models in this application are **anomaly detection models** that predict whether network traffic is **safe (normal)** or **unsafe (anomalous/malicious)**.

### Prediction Output

Each model predicts a binary classification:
- **0 = Safe (Normal)** - Legitimate network traffic
- **1 = Unsafe (Anomaly)** - Potentially malicious or anomalous network traffic

### Prediction Format

When you make a prediction, the model returns:

```json
{
  "prediction": 0,                    // 0 = safe, 1 = unsafe
  "label": "safe",                    // Human-readable label
  "probability_safe": 0.95,          // Probability of being safe (0-1)
  "probability_unsafe": 0.05,        // Probability of being unsafe (0-1)
  "confidence": 0.95                  // Confidence in the prediction (max probability)
}
```

## Model Types

The application supports three types of anomaly detection models:

### 1. **RFv1 - Random Forest Classifier** (Supervised)
- **Type**: Supervised learning
- **Training**: Requires labeled data (0 = safe, 1 = unsafe)
- **How it works**: 
  - Trains on labeled examples to learn patterns
  - Uses ensemble of decision trees
  - Provides probability scores for both classes
- **Output**: Binary classification with probabilities

### 2. **IFv1 - Isolation Forest** (Unsupervised)
- **Type**: Unsupervised learning
- **Training**: Can work with unlabeled data (learns what's "normal")
- **How it works**:
  - Identifies outliers by isolating them in feature space
  - Uses contamination parameter (expected proportion of anomalies)
  - Returns anomaly scores (higher = more anomalous)
- **Output**: Converts to binary (0 = inlier/normal, 1 = outlier/anomaly)

### 3. **AEv1 - Autoencoder** (Unsupervised)
- **Type**: Unsupervised learning (neural network)
- **Training**: Learns to reconstruct normal traffic patterns
- **How it works**:
  - Trains to compress and reconstruct input data
  - High reconstruction error = anomaly
  - Uses reconstruction error threshold
- **Output**: Converts reconstruction error to binary classification

## Features Used for Prediction

The models use **network traffic flow features** extracted from network packets. These features are divided into several categories:

### 1. **Basic Flow Statistics** (44 features)

#### Duration and Packet Counts
- `dur` - Flow duration (seconds)
- `Spkts` - Source-to-destination packet count
- `Dpkts` - Destination-to-source packet count
- `sbytes` - Source-to-destination byte count
- `dbytes` - Destination-to-source byte count

#### Rate and Load
- `rate` - Flow rate (packets per second)
- `Sload` - Source bits per second
- `Dload` - Destination bits per second

#### Time-to-Live (TTL)
- `sttl` - Source TTL value
- `dttl` - Destination TTL value

#### Packet Loss
- `sloss` - Source packets retransmitted or dropped
- `dloss` - Destination packets retransmitted or dropped

#### Inter-Packet Timing
- `Sintpkt` - Source inter-packet arrival time (microseconds)
- `Dintpkt` - Destination inter-packet arrival time (microseconds)
- `Sjit` - Source jitter (variation in inter-packet time)
- `Djit` - Destination jitter

#### TCP Window and Buffer
- `swin` - Source TCP window advertisement value
- `dwin` - Destination TCP window advertisement value
- `stcpb` - Source TCP base sequence number
- `dtcpb` - Destination TCP base sequence number

#### TCP Round-Trip Time
- `tcprtt` - TCP connection setup round-trip time
- `synack` - TCP SYN-ACK time
- `ackdat` - TCP ACK to data packet time

#### Packet Sizes
- `smeansz` - Mean of the flow packet size transmitted by the source
- `dmeansz` - Mean of the flow packet size transmitted by the destination

#### HTTP/Application Layer
- `trans_depth` - Represents the pipelined depth into the connection
- `res_bdy_len` - Actual uncompressed content size of the data transferred

### 2. **Connection Tracking Features** (10 features)

These features track connection patterns and state:

- `ct_srv_src` - No. of connections that contain the same service and source address
- `ct_state_ttl` - No. of connections for each state according to specific value of TTL
- `ct_dst_ltm` - No. of connections of the same destination address in 100ms
- `ct_src_dport_ltm` - No. of connections of the same source address and destination port in 100ms
- `ct_dst_sport_ltm` - No. of connections of the same destination address and source port in 100ms
- `ct_dst_src_ltm` - No. of connections of the same destination address and source address in 100ms
- `ct_src_ltm` - No. of connections of the same source address in 100ms
- `ct_srv_dst` - No. of connections that contain the same service and destination address

### 3. **Protocol-Specific Features** (4 features)

- `is_ftp_login` - 1 if the FTP session is accessed by user and password; 0 otherwise
- `ct_ftp_cmd` - No. of flows that has a command in FTP session
- `ct_flw_http_mthd` - No. of flows using HTTP methods (GET, POST, etc.)
- `is_sm_ips_ports` - 1 if source and destination IPs and ports match; 0 otherwise

### 4. **Derived Ratio Features** (4 features)

- `byte_ratio` - Ratio of source bytes to destination bytes
- `pkt_ratio` - Ratio of source packets to destination packets
- `flow_rate` - Flow rate (flows per second)
- `pkt_rate` - Packet rate (packets per second)

### 5. **One-Hot Encoded Protocol Features** (~150+ features)

Binary features for each network protocol:
- `proto_tcp`, `proto_udp`, `proto_icmp`, etc.
- Each protocol gets its own binary feature (1 if used, 0 otherwise)
- Examples: `proto_tcp`, `proto_udp`, `proto_http`, `proto_https`, etc.

### 6. **One-Hot Encoded State Features** (7 features)

TCP connection states:
- `state_CON` - Connection established
- `state_INT` - Connection initialization
- `state_FIN` - Connection finished
- `state_RST` - Connection reset
- `state_CLO` - Connection closed
- `state_ACC` - Connection accepted
- `state_REQ` - Connection requested

### 7. **One-Hot Encoded Service Features** (12 features)

Network services:
- `service_http`, `service_https`, `service_ftp`, `service_ssh`, `service_dns`, etc.
- Each service gets its own binary feature

## Total Feature Count

The models typically use **~196 features** total:
- 44 basic flow statistics
- 10 connection tracking features
- 4 protocol-specific features
- 4 ratio features
- ~150+ protocol one-hot encoded features
- 7 state one-hot encoded features
- 12 service one-hot encoded features

## Excluded Fields

The following fields are **NOT** used as features (they are metadata or the target variable):

- `label` - The target variable (what we're predicting) - **EXCLUDED**
- `id` - Record identifier - **EXCLUDED**
- `attack_cat` - Attack category (metadata) - **EXCLUDED**
- `upload_timestamp` - Timestamp (metadata) - **EXCLUDED**
- `T` - Training/testing split indicator - **EXCLUDED**

## How Features Are Used

1. **Feature Extraction**: All numeric features are extracted from network flow data
2. **Feature Selection**: The model uses all features except excluded metadata fields
3. **Preprocessing**: 
   - Missing values are filled with 0
   - Non-numeric values are converted to numeric
   - Features are standardized/normalized for some models
4. **Prediction**: The model uses all features together to make a prediction

## Example Prediction Request

```json
{
  "data": [
    {
      "dur": 0.5,
      "Spkts": 10,
      "Dpkts": 8,
      "sbytes": 1024,
      "dbytes": 512,
      "rate": 20.0,
      "sttl": 64,
      "dttl": 64,
      "Sload": 16384.0,
      "Dload": 8192.0,
      "proto_tcp": 1,
      "proto_udp": 0,
      "state_CON": 1,
      "service_http": 1,
      // ... all other features
    }
  ]
}
```

## Model Training

During training:
- Models learn patterns from historical network traffic
- They identify which feature combinations indicate anomalies
- Random Forest can show feature importance (which features matter most)
- Models are saved with their feature names to ensure consistency

## Real-World Application

This system is designed for **campus IoT network security**:
- Monitors network traffic in real-time
- Detects anomalies that could indicate:
  - Malware infections
  - DDoS attacks
  - Unauthorized access attempts
  - Data exfiltration
  - Botnet activity
  - Other security threats

The models analyze network flow characteristics to identify traffic that deviates from normal patterns, helping security teams detect and respond to threats quickly.
