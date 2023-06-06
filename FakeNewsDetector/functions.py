import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


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
    
