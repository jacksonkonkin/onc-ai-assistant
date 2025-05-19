# onc-ai-assistant

## Steps to run llama3 and rag pipeline

1. Install ollama and download llama3.1

```
brew install ollama

ollama serve

ollama pull llama3.1
```



1. Install dependencies

```pip install -r requirements.txt```

2. Set API key

```export OPENAI_API_KEY=your_key_here```

3. Run with your JSON file

```python generic_rag.py --data your_data.json```