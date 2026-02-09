import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Flipkart Sentiment Analyzer")

# ===============================
# Download NLTK (only once)
# ===============================
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

# ===============================
# Load Model + Vectorizer (CACHED)
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ===============================
# Text Preprocessing
# ===============================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ===============================
# Streamlit UI
# ===============================
import streamlit as st

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ===============================
# CUSTOM CSS (COLORFUL & MODERN)
# ===============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

.main {
    background: transparent;
}

.glass-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.15);
    margin-top: 30px;
}

.title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    background: linear-gradient(90deg, #2874F0, #ff9800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.stTextArea textarea {
    border-radius: 14px;
    font-size: 16px;
    border: 2px solid #e0e0e0;
}

.stButton button {
    width: 100%;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #2874F0, #ff9800);
    color: white;
    border: none;
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}

.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    animation: pop 0.4s ease;
}

.positive {
    background: linear-gradient(135deg, #43cea2, #185a9d);
    color: white;
}

.negative {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
}

@keyframes pop {
    0% { transform: scale(0.9); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #black;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# UI LAYOUT
# ===============================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🛒 Flipkart Review Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Understand customer emotions instantly using Machine Learning</div>", unsafe_allow_html=True)

review = st.text_area(
    "✍️ Enter Product Review",
    height=150,
    placeholder="Example: Waste of money, very bad quality..."
)

predict = st.button("🔍 Analyze Sentiment")

if predict:
    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]

        if prediction == 1:
            st.markdown(
                "<div class='result positive'>✅ Positive Review 😊</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result negative'>❌ Negative Review 😞</div>",
                unsafe_allow_html=True
            )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>Built with ❤️ using Machine Learning & Streamlit</div>",
    unsafe_allow_html=True
)
