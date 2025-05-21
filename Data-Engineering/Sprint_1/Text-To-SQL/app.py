"""
Vanna + Ollama SQLite Sales Data Query Tool

This script demonstrates loading sales data from a CSV file into an SQLite database,
then using Vanna combined with a local Ollama language model (llama3.2:latest) to
train on the database schema and answer natural language questions about the data.
It also optionally exposes a Flask API to query the system via HTTP.

---

Features:
- Load CSV sales data into an SQLite database.
- Extract and train on database schema (DDL).
- Add domain knowledge and example queries.
- Use Ollama LLM locally to convert natural language to SQL.
- Retrieve answers from the database and return responses.
- Optionally expose a Flask web API.

---

Requirements:
- Python 3.8+
- pandas
- sqlite3 (built-in)
- vanna
- Ollama installed locally with model llama3.2:latest
- Flask (optional, for API)
- sales.csv file with columns: year, region, revenue

---

Installation:
1. Create and activate virtualenv (optional):
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies:
   pip install pandas vanna flask httpx

3. Install and run Ollama locally, and confirm model with:
   ollama list

4. Place sales.csv in the project directory.

---

Usage:
Run the script:
python app.py

This loads CSV into financedb.db, trains Vanna, runs a sample query,
and starts a Flask API server on http://localhost:5000

---

Troubleshooting:
- Install missing packages like httpx if needed.
- Make sure Ollama daemon is running.
- Verify CSV file path and format.
- Use Python 3.8+.

"""

import sqlite3
import pandas as pd

# Load CSV data
df = pd.read_csv('./sales.csv')

# Connect to SQLite DB (will create if not exists)
conn = sqlite3.connect('financedb.db')

# Create or replace 'sales' table with CSV data
df.to_sql('sales', conn, if_exists='replace', index=False)

# Read from 'sales' table to verify
df_from_db = pd.read_sql_query("SELECT * FROM sales", conn)
print("Data in sales table:")
print(df_from_db)

# Close connection (Vanna will reconnect)
conn.close()


# # --- Vanna setup ---

from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model': 'llama3.2:latest'})

# Connect to the same SQLite DB
vn.connect_to_sqlite('financedb.db')

# Get DDL for all tables
df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL")

# Train Vanna with all table definitions
for ddl in df_ddl['sql'].to_list():
    vn.train(ddl=ddl)

# Add some domain knowledge or documentation
vn.train(documentation="This database contains sales data by year, region, and revenue.")

# Add example query training (adjust as you want)
vn.train(sql="SELECT * FROM sales WHERE region='North'")

# Inspect training data if you want
training_data = vn.get_training_data()
print("Training data:", training_data)

# Ask a sample question
response = vn.ask(question="What is the total revenue for the North region?")
print("Vanna answer:", response)


# --- Optional: Run Flask app (if you want to expose as API) ---

from vanna.flask import VannaFlaskApp

app = VannaFlaskApp(vn)

if __name__ == '__main__':
    app.run()
