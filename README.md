# Text Summarization Project

This project implements a basic text summarization pipeline using Python. It includes preprocessing, indexing, word counting, and displaying top documents through a simple GUI. It also utilizes an inverted index for efficient document lookup.

## 📁 Project Structure

- `gui.py`: GUI interface for user interaction.
- `indexing.py`: Handles the indexing of documents.
- `inverted_index.py`: Code to create an inverted index from the preprocessed documents.
- `inverted_index.csv`: Stores the generated inverted index in CSV format.
- `preprocess.py`: Performs preprocessing on the input text (e.g., removing stopwords, tokenization).
- `top_documents.py`: Finds and displays the top documents based on some criteria (e.g., word count, relevance).
- `wordcount_after_preprocess.py`: Counts word frequency after preprocessing.
- `README.md`: This file.

## 🧰 Features

- Document preprocessing (cleaning, tokenizing, etc.)
- Inverted index generation
- Word count tracking
- Ranking documents
- Simple GUI to interact with the application

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Required packages:
  ```bash
  pip install nltk pandas tkinter


