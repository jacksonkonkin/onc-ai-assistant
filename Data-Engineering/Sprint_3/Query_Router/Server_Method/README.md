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


## 4. Question Types

| Query Type               | Description                                                                                                         | Example                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| general_knowledge        | Asks for definitions, explanations, or scientific background—not for data or stats.                                 | What is ocean salinity? <br> Explain how a radiometer works.                                              |
| data_discovery           | Discovers what devices, sensors, or data types are available at a location.                                         | What sensors are located at CBYIP? <br> Which parameters can be measured at Cambridge Bay?                |
| data_download_instance   | Requests the value of a parameter at a specific location and time—usually a single reading.                         | Give me the temperature in CBY on June 12, 2024. <br> What was the oxygen level at CBYIP on March 10, 2023?|
| data_download_interval   | Requests statistical summaries (min, max, average, count, etc.) over a time interval or relative period.            | What is the average temperature at CBY between June 1 and June 30, 2024? <br> Show me the max salinity at CF240 in July 2023.|
| data_and_knowledge_query | Requires both domain/scientific understanding and use of real data/statistics—applied, interpretive, or advisory.  | Is it safe to travel on the ice at Cambridge Bay during March 2024 based on average ice thickness data?<br>Does the oxygen level at CBYIP in July suggest healthy conditions for fish?|
