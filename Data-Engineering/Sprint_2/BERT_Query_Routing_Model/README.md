
# BERT Query Classifier

This project uses a fine-tuned BERT model to classify natural language queries into predefined categories using HuggingFace's `transformers` library.

## Simple Instructions

1. Make sure you have Python 3.7 or above installed.

2. Install the required libraries by running:
   ```
   pip install torch transformers scikit-learn
   ```

3. Place the following files in the same folder:
   - `classify_query.py` (the main script)
   - `bert_query_classifier/` (folder with BERT model files: `pytorch_model.bin`, `config.json`, etc.)
   - `label_encoder.pkl` (label encoder used during training)

4. Open your terminal and navigate to the folder.

5. Run the script:
   ```
   python classify_query.py
   ```

6. Enter a query when prompted and press Enter.

7. The script will print the predicted label.

## Example

```
Enter a query: book a flight to Toronto
Predicted label: travel
```
