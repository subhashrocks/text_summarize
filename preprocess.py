import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
