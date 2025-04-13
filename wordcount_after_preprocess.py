import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the preprocessed dataset
preprocessed_file_path = 'preprocessed_data.csv'
df = pd.read_csv(preprocessed_file_path)

# Replace NaN values in 'cleaned_review' column with empty strings
df['cleaned_review'] = df['cleaned_review'].fillna('')

# Word Cloud to visualize frequent words
all_text = ' '.join(df['cleaned_review'])
wordcloud = WordCloud(width=800, height=400, random_state=42, max_font_size=100, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Preprocessed Text Data')
plt.show()

# Sentiment Analysis
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['cleaned_review'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Visualize sentiment distribution
plt.figure(figsize=(8, 5))
df['sentiment'].plot(kind='hist', bins=30, color='skyblue', edgecolor='black')
plt.title('Sentiment Distribution of Text Data')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Frequency')
plt.show()
