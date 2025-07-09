# Query Classifier API

A lightweight FastAPI web service that classifies user queries using a pre-trained Sentence Transformer (`all-MiniLM-L6-v2`). It loads labeled questions from multiple CSV files and returns the most similar label for a given query.


## 1. Install Dependencies

```bash
pip install fastapi uvicorn pandas sentence-transformers torch
```

NOTE: Make sure to keep the csv files along with script as they are needed when you run the model for the first time at least.

### Project Structure:

├── app.py # Main script
├── deployments.csv
├── device_info.csv
├── property.csv
├── device_category.csv
├── locations.csv
├── deployments_2.csv
├── device_info_2.csv
├── property_2.csv
├── device_category_2.csv
├── locations_2.csv
├── data_and_knowledge_queries.csv

## 2. Run the API

```bash
uvicorn app:app --reload
```

## 3. How to use it

```bash
curl -X POST http://localhost:8000/classify \
     -H "Content-Type: application/json" \
     -d '{"query": "How many devices are deployed in building B?"}'
```

It returns:

```bash
{
  "label": "data_discovery"
}
```