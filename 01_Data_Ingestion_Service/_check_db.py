import sqlite3

conn = sqlite3.connect("campus_iot_data.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
tables = [r[0] for r in cursor.fetchall()]
print(f"Found {len(tables)} tables:")
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM [{t}]")
    count = cursor.fetchone()[0]
    print(f"  {t}: {count} rows")
conn.close()
