import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

#Reading the data
store_data = pd.read_csv('C:/Users/Rishab/Downloads/news.csv')
#Get shape and head
store_data.shape
store_data.head()

labels=store_data.label
labels.head()
x_train,x_test,y_train,y_test=train_test_split(store_data['text'], labels, test_size=0.2, random_state=7)
#Initializing a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transforming training set, transforming test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predicting on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
# print(f'Accuracy: {round(score*100,2)}%')
# print(labels)


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
def preprocess():
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z]", " ", text)
    
    # Tokenize the text into individual words
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a single string
    preprocessed_text = " ".join(words)
    
    return preprocessed_text
    
    
    
x=input("Enter your news article: ")
news_article = x
preprocessed_article = preprocess(news_article)

# Transform the preprocessed news article
tfidf_article = tfidf_vectorizer.transform([preprocessed_article])

# Predict the label using the trained model
predicted_label = pac.predict(tfidf_article)

print(predicted_label)