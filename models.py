# models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import hashlib

# --- Flow schema (partial; include fields you need) ---
class FlowRecord(BaseModel):
    flow_id: str = Field(..., alias="Flow ID")
    src_ip: Optional[str] = Field(None, alias="Src IP")
    src_port: Optional[int] = Field(None, alias="Src Port")
    dst_ip: Optional[str] = Field(None, alias="Dst IP")
    dst_port: Optional[int] = Field(None, alias="Dst Port")
    protocol: Optional[str] = Field(None, alias="Protocol")
    timestamp: Optional[str] = Field(None, alias="Timestamp")
    flow_duration: Optional[float] = Field(None, alias="Flow Duration")
    total_fwd_packets: Optional[int] = Field(None, alias="Total Fwd Packet")
    total_bwd_packets: Optional[int] = Field(None, alias="Total Bwd packets")
    flow_bytes_s: Optional[float] = Field(None, alias="Flow Bytes/s")
    flow_packets_s: Optional[float] = Field(None, alias="Flow Packets/s")
    flow_iat_mean: Optional[float] = Field(None, alias="Flow IAT Mean")
    fwd_iat_mean: Optional[float] = Field(None, alias="Fwd IAT Mean")
    bwd_iat_mean: Optional[float] = Field(None, alias="Bwd IAT Mean")
    packet_length_mean: Optional[float] = Field(None, alias="Packet Length Mean")
    # ... add others as needed
    label: Optional[int] = Field(None, alias="Label")

    @validator("src_port", "dst_port", pre=True)
    def to_int_or_none(cls, v):
        if v is None or v == "":
            return None
        return int(v)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Return a flattened dict with normalized keys for storage."""
        d = self.dict(by_alias=False)
        # unify keys to snake_case (crudely)
        out = {}
        for k, v in d.items():
            out[k.lower()] = v
        # add a derived feature: bytes_per_packet if possible
        total_pkts = (self.total_fwd_packets or 0) + (self.total_bwd_packets or 0)
        if total_pkts and self.flow_bytes_s:
            out["bytes_per_packet"] = (self.flow_bytes_s) / total_pkts
        else:
            out["bytes_per_packet"] = None
        # compact hashing for high-card fields
        if self.src_ip:
            out["src_ip_hash"] = simple_hash(self.src_ip)
        if self.dst_ip:
            out["dst_ip_hash"] = simple_hash(self.dst_ip)
        if self.protocol:
            out["protocol_hash"] = simple_hash(self.protocol)
        return out

def simple_hash(value: str, n_bits: int = 16) -> int:
    """Stable integer hash for categorical -> small cardinality."""
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(h, 16) % (2**n_bits)


# --- Behavior schema (partial) ---
class BehaviorRecord(BaseModel):
    stream: Optional[str] = Field(None, alias="stream")
    device_mac: Optional[str] = Field(None, alias="(device_mac) Label 1 for DI")
    src_ip: Optional[str] = Field(None, alias="src_ip")
    dst_ip: Optional[str] = Field(None, alias="dst_ip")
    inter_arrival_time: Optional[float] = Field(None, alias="inter_arrival_time")
    payload_entropy: Optional[float] = Field(None, alias="payload_entropy")
    eth_size: Optional[int] = Field(None, alias="eth_size")
    ttl: Optional[int] = Field(None, alias="ttl")
    http_request_method: Optional[str] = Field(None, alias="http_request_method")
    http_response_code: Optional[str] = Field(None, alias="http_response_code")
    user_agent: Optional[str] = Field(None, alias="User_Agent")
    payload_length: Optional[int] = Field(None, alias="payload_length")
    highest_layer: Optional[str] = Field(None, alias="highest_layer")
    # add more fields as necessary...
    label_ad: Optional[int] = Field(None, alias="Label 2 for AD")

    def to_flat_dict(self) -> Dict[str, Any]:
        d = self.dict(by_alias=False)
        out = {}
        for k, v in d.items():
            out[k.lower()] = v
        if self.device_mac:
            out["device_mac_hash"] = simple_hash(self.device_mac)
        if self.src_ip:
            out["src_ip_hash"] = simple_hash(self.src_ip)
        return out
