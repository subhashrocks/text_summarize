import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


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

cosine_similarities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

# Sort cosine similarities in descending order and get corresponding document IDs
sorted_cosine_similarities = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)
sorted_document_ids = [doc_id for doc_id, _ in sorted_cosine_similarities]

# Initialize lists to store precision and recall values
precision_values = []
recall_values = []

# Initialize variables to keep track of true positives and retrieved documents
true_positives = 0
retrieved_documents = 0

# Calculate precision and recall for different thresholds
for doc_id in sorted_document_ids:
    retrieved_documents += 1
    if relevance_feedback[doc_id] == 1:
        true_positives += 1
    precision = true_positives / retrieved_documents
    recall = true_positives / sum(relevance_feedback)
    precision_values.append(precision)
    recall_values.append(recall)

# Plot the PR curve
plt.figure()
plt.plot(recall_values, precision_values, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.grid()
plt.show()
