import tkinter as tk
from tkinter import messagebox, Label, Text, Button, Entry, END
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Function for text preprocessing
def preprocess_text(text):
    # Remove irrelevant characters, symbols, and formatting issues
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Ensure text is converted to string

    # Tokenize the text into words
    tokens = word_tokenize(text.lower())  # Convert to lowercase

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# Function to perform search and display results
def perform_search():
    query = query_entry.get()
    if not query:
        messagebox.showinfo("Error", "Please enter a query.")
        return

    query_tokens = [word for word in word_tokenize(query.lower()) if word not in stopwords.words('english')]
    query_text = ' '.join(query_tokens)

    # Load the preprocessed dataset
    preprocessed_file_path = 'preprocessed_data.csv'
    df = pd.read_csv(preprocessed_file_path)

    # Replace NaN values in 'cleaned_review' column with empty string
    df['cleaned_review'] = df['cleaned_review'].fillna('')

    # Use TfidfVectorizer to calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['cleaned_review'].values.tolist() + [query_text])

    # Calculate cosine similarity
    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()

    # Get the top 10 documents based on cosine similarity
    top_documents = cosine_similarities.argsort()[:-11:-1]

    results_text.delete(1.0, END)
    results_text.insert(END, "Top 10 Documents:\n")
    for rank, doc_id in enumerate(top_documents, 1):
        results_text.insert(END, f"Rank {rank}: Document {doc_id}, Similarity Score: {cosine_similarities[doc_id]}\n")
        results_text.insert(END, df.loc[doc_id, 'cleaned_review'] + '\n')
        results_text.insert(END, '-' * 50 + '\n')

# Create the main window
window = tk.Tk()
window.title("Text Search Engine")

# Create labels and text entry for query
query_label = Label(window, text="Enter your query:")
query_label.pack()
query_entry = Entry(window, width=50)
query_entry.pack()

# Create a button to perform the search
search_button = Button(window, text="Search", command=perform_search)
search_button.pack()

# Create a text box for displaying search results
results_text = Text(window, width=80, height=20)
results_text.pack()

# Start the GUI event loop
window.mainloop()
