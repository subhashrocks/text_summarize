import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'Reviews_new.csv'
df = pd.read_csv(dataset_path, low_memory=False)

# Update column names based on your dataset structure
text_column_name = 'Text'

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

# Apply the preprocessing function to the 'Text' column
if text_column_name in df.columns:
    df['cleaned_review'] = df[text_column_name].apply(preprocess_text)
else:
    print(f"Error: '{text_column_name}' column not found in the dataset.")

# Save the preprocessed data to a new CSV file
output_file_path = 'preprocessed_data.csv'
df.to_csv(output_file_path, index=False)

print(f"Preprocessed data saved to '{output_file_path}'.")

# Load the preprocessed dataset
preprocessed_file_path = 'preprocessed_data.csv'
df = pd.read_csv(preprocessed_file_path)

# Replace NaN values in 'cleaned_review' column with empty string
df['cleaned_review'] = df['cleaned_review'].fillna('')

# Create a defaultdict to store the inverted index
inverted_index = defaultdict(set)

# Tokenize and build the inverted index
for doc_id, review in enumerate(df['cleaned_review']):
    tokens = word_tokenize(review.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    for term in set(filtered_tokens):  # Use set to avoid duplicate entries for a term in a document
        inverted_index[term].add(doc_id)

# Convert sets to lists for easier serialization (e.g., saving to a file)
inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

# Save the inverted index to a file (e.g., CSV or JSON)
output_file_path = 'inverted_index.csv'
inverted_index_df = pd.DataFrame(list(inverted_index.items()), columns=['Term', 'DocumentIDs'])
inverted_index_df.to_csv(output_file_path, index=False)

print(f"Inverted index saved to '{output_file_path}'.")

# Load the preprocessed dataset
preprocessed_file_path = 'preprocessed_data.csv'
df = pd.read_csv(preprocessed_file_path)

# Replace NaN values in 'cleaned_review' column with empty string
df['cleaned_review'] = df['cleaned_review'].fillna('')

# Tokenize the query
query = "vegetarian recipes with quinoa"
query_tokens = [word for word in word_tokenize(query.lower()) if word not in stopwords.words('english')]
query_text = ' '.join(query_tokens)

# Use TfidfVectorizer to calculate TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['cleaned_review'].values.tolist() + [query_text])

# Calculate cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()

# Get the top 10 documents based on cosine similarity
top_documents = cosine_similarities.argsort()[:-11:-1]

# Print the results
print("Top 10 Documents:")
for rank, doc_id in enumerate(top_documents, 1):
    print(f"Rank {rank}: Document {doc_id}, Similarity Score: {cosine_similarities[doc_id]}")
    print(df.loc[doc_id, 'cleaned_review'])
    print('-' * 50)

# Ask the user to provide relevance feedback (1 for relevant, 0 for non-relevant)
relevance_feedback = []
for doc_id in top_documents:
    feedback = int(input(f"Is document {doc_id} relevant? (1 for relevant, 0 for non-relevant): "))
    relevance_feedback.append(feedback)

# Filter and print relevant documents
relevant_documents = [doc_id for doc_id, feedback in zip(top_documents, relevance_feedback) if feedback == 1]

if relevant_documents:
    print("\nRelevant Documents:")
    for doc_id in relevant_documents:
        print(f"Document {doc_id}, Similarity Score: {cosine_similarities[doc_id]}")
        print(df.loc[doc_id, 'cleaned_review'])
        print('-' * 50)
else:
    print("\nNo relevant documents based on user feedback.")
