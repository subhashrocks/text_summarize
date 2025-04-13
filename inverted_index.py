import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import re

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


# Load the dataset
dataset_path = 'reviews_new.csv'
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
