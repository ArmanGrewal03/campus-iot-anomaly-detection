# storage.py
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Text
from sqlalchemy.sql import insert
import pandas as pd

DB_URL = "sqlite:///ingest_poc.db"

engine = create_engine(DB_URL, echo=False)
meta = MetaData()

# Minimal flow table (flexible)
flow_table = Table(
    "flows", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("flow_id", String(128)),
    Column("src_ip_hash", Integer),
    Column("dst_ip_hash", Integer),
    Column("protocol_hash", Integer),
    Column("flow_duration", Float),
    Column("total_fwd_packets", Integer),
    Column("total_bwd_packets", Integer),
    Column("flow_bytes_s", Float),
    Column("bytes_per_packet", Float),
    Column("packet_length_mean", Float),
    Column("label", Integer),
    Column("raw_json", Text),
)

behavior_table = Table(
    "behavior", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("device_mac_hash", Integer),
    Column("src_ip_hash", Integer),
    Column("payload_entropy", Float),
    Column("eth_size", Integer),
    Column("ttl", Integer),
    Column("highest_layer", String(64)),
    Column("label_ad", Integer),
    Column("raw_json", Text),
)

meta.create_all(engine)

def store_flow_record(flat: dict, raw_json: dict):
    ins = flow_table.insert().values(
        flow_id=flat.get("flow_id"),
        src_ip_hash=flat.get("src_ip_hash"),
        dst_ip_hash=flat.get("dst_ip_hash"),
        protocol_hash=flat.get("protocol_hash"),
        flow_duration=flat.get("flow_duration"),
        total_fwd_packets=flat.get("total_fwd_packets"),
        total_bwd_packets=flat.get("total_bwd_packets"),
        flow_bytes_s=flat.get("flow_bytes_s"),
        bytes_per_packet=flat.get("bytes_per_packet"),
        packet_length_mean=flat.get("packet_length_mean"),
        label=flat.get("label"),
        raw_json=str(raw_json)
    )
    with engine.connect() as conn:
        conn.execute(ins)

def store_behavior_record(flat: dict, raw_json: dict):
    ins = behavior_table.insert().values(
        device_mac_hash=flat.get("device_mac_hash"),
        src_ip_hash=flat.get("src_ip_hash"),
        payload_entropy=flat.get("payload_entropy"),
        eth_size=flat.get("eth_size"),
        ttl=flat.get("ttl"),
        highest_layer=flat.get("highest_layer"),
        label_ad=flat.get("label_ad"),
        raw_json=str(raw_json)
    )
    with engine.connect() as conn:
        conn.execute(ins)
