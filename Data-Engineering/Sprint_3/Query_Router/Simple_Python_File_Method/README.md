# Sentence Classifier (Terminal Version)

This is a **console-based semantic query classifier** built with Python and [Sentence Transformers](https://www.sbert.net/). It classifies user queries into predefined labels by comparing them against labeled questions from multiple CSV files using **cosine similarity**.

It’s designed to run **interactively in terminal**, keeping the model and data **loaded in memory** for fast, repeated use until manually exit.

---

## Question Types

| Query Type               | Description                                                                                                         | Example                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| general_knowledge        | Asks for definitions, explanations, or scientific background—not for data or stats.                                 | What is ocean salinity? <br> Explain how a radiometer works.                                              |
| data_discovery           | Discovers what devices, sensors, or data types are available at a location.                                         | What sensors are located at CBYIP? <br> Which parameters can be measured at Cambridge Bay?                |
| data_download_instance   | Requests the value of a parameter at a specific location and time—usually a single reading.                         | Give me the temperature in CBY on June 12, 2024. <br> What was the oxygen level at CBYIP on March 10, 2023?|
| data_download_interval   | Requests statistical summaries (min, max, average, count, etc.) over a time interval or relative period.            | What is the average temperature at CBY between June 1 and June 30, 2024? <br> Show me the max salinity at CF240 in July 2023.|
| data_and_knowledge_query | Requires both domain/scientific understanding and use of real data/statistics—applied, interpretive, or advisory.  | Is it safe to travel on the ice at Cambridge Bay during March 2024 based on average ice thickness data?<br>Does the oxygen level at CBYIP in July suggest healthy conditions for fish?|


---

## Features

- Loads data and model only once
- Works entirely in the terminal — no API, no web server
- Uses `all-MiniLM-L6-v2` for fast and accurate semantic similarity
- Supports multiple CSV sources with `question` and `label` columns
- Clean shutdown with `Ctrl+C`, `exit`, or `quit`

---

## Dependencies


```bash
pip install pandas torch sentence-transformers
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
## To Run

```bash
python sentence_classifier.py
```
