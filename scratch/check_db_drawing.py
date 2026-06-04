import sqlite3
conn = sqlite3.connect("neuroscan.db")
cursor = conn.cursor()
cursor.execute("SELECT timestamp, test_type, result FROM patient_history WHERE test_type='drawing'")
for row in cursor.fetchall():
    print(f"Timestamp: {row[0]}, Type: {row[1]}, Result: {row[2]}")
conn.close()
