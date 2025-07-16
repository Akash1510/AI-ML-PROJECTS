import streamlit as st # type: ignore
import pickle
import nltk # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore

# Download NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load Model and TF-IDF Vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing Function
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.set_page_config(page_title="Restaurant Review Sentiment", page_icon="🍽️")
st.title("🍽️ Restaurant Review Sentiment Analyzer")

user_input = st.text_area("Enter your restaurant review:",placeholder="Type your review here...")

if st.button("Analyze"):
    clean_text = preprocess(user_input)
    vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized)[0]
    # to show the confidence score
    confidence = model.predict_proba(vectorized)[0][prediction]
    st.write(f"Confidence Score: {confidence*100:.2f}% ")
  


    if prediction == 1:
        st.success("✅ Positive Review!")
    else:
        st.error("❌ Negative Review!")
