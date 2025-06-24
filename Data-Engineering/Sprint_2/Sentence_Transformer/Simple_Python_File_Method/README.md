# Sentence Classifier (Terminal Version)

This is a **console-based semantic query classifier** built with Python and [Sentence Transformers](https://www.sbert.net/). It classifies user queries into predefined labels by comparing them against labeled questions from multiple CSV files using **cosine similarity**.

It’s designed to run **interactively in your terminal**, keeping the model and data **loaded in memory** for fast, repeated use until you manually exit.

---

## Features

- Loads your data and model only once
- Works entirely in the terminal — no API, no web server
- Uses `all-MiniLM-L6-v2` for fast and accurate semantic similarity
- Supports multiple CSV sources with `question` and `label` columns
- Clean shutdown with `Ctrl+C`, `exit`, or `quit`

---

## Dependencies


```bash
pip install pandas torch sentence-transformers
```

## To Run

```bash
python sentence_classifier.py
```
