import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from wordcloud import WordCloud

#I have done this in google colab, make sure you give your file path correctly.
cols=['ID', 'Topic', 'Sentiment', 'Text']
train = pd.read_csv(r"/content/twitter_training.csv",names=cols)
train.head()
train.shape
train.info()
train.describe(include=object)
train['Sentiment'].unique()
train.isnull().sum()
train.dropna(inplace=True)
train.isnull().sum()
train.duplicated().sum()
train.drop_duplicates(inplace=True)
train.duplicated().sum()

import matplotlib.pyplot as plt
# Set the background color for the plot
plt.style.use('dark_background')
# Create the plot
plt.figure(figsize=(8, 10))
train['Topic'].value_counts().plot(kind='barh', color='g')
# Set labels and display the plot
plt.xlabel("Count")
plt.show()

sns.countplot(x = 'Sentiment',data=train,palette='viridis')
plt.show()

# Calculate the counts for each sentiment
sentiment_counts = train['Sentiment'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140, colors=['skyblue', 'orange', 'green', 'red', 'purple'])

plt.title('Sentiment Distribution')

# Show the plot
plt.show()
train
plt.figure(figsize=(20,12))
sns.countplot(x='Topic',data=train,palette='viridis',hue='Sentiment')
plt.xticks(rotation=90)
plt.show()

## Group by Topic and Sentiment
topic_wise_sentiment = train.groupby(["Topic", "Sentiment"]).size().reset_index(name='Count')

# Step 2: Select Top 5 Topics
topic_counts = train['Topic'].value_counts().nlargest(5).index
top_topics_sentiment = topic_wise_sentiment[topic_wise_sentiment['Topic'].isin(topic_counts)]
plt.figure(figsize=(12, 8))
sns.barplot(data=top_topics_sentiment[top_topics_sentiment['Sentiment'] == 'Negative'], x='Topic', y='Count', palette='viridis')
plt.title('Top 5 Topics with Negative Sentiments')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(data=top_topics_sentiment[top_topics_sentiment['Sentiment'] == 'Positive'], x='Topic', y='Count', palette='Greens')
plt.title('Top 5 Topics with Positive Sentiments')
plt.xlabel('Topic')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()