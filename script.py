import nltk
import streamlit as st
import string
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download()
#nltk.download('averaged_perception_tagger')


def preprocess(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned_sentences = []
    for sent in sentences:
        words = word_tokenize(sent.lower())  # Convert to lowercase and tokenize
        words = [word for word in words if word.isalnum() and word not in stop_words]
        cleaned_sentences.append(" ".join(words))  # Join words back into sentences
    
    return sentences, cleaned_sentences

def get_most_relevant_sentence(user_query, original_sentences, cleaned_sentences):
    cleaned_query = " ".join(
        [word for word in word_tokenize(user_query.lower()) if word.isalnum()]
    )
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([cleaned_query] + cleaned_sentences)
     # Compute cosine similarity
    similarity_scores = cosine_similarity(vectors[0], vectors[1:])[0]

    # Get the most relevant sentence
    best_match_index = np.argmax(similarity_scores)
    return original_sentences[best_match_index]


# Define chatbot function
def chatbot(user_query):
    file_path = "data.txt"  # Change this to your text file name
    original_sentences, cleaned_sentences = preprocess(file_path)

    if not user_query.strip():
        return "Please enter a question."

    return get_most_relevant_sentence(user_query, original_sentences, cleaned_sentences)

