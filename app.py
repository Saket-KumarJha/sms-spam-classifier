import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)



# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="🛡️",
    layout="centered"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e1b4b);
}

.block-container {
    max-width: 800px;
    padding-top: 50px;
    padding-bottom: 40px;
}

.header {
    text-align: center;
    margin-bottom: 35px;
}

.header-icon {
    font-size: 48px;
    margin-bottom: 8px;
}

.header-title {
    font-size: 38px;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 16px;
    color: #94a3b8;
}

.input-title {
    font-size: 21px;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 12px;
}

textarea {
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #475569 !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    padding: 15px !important;
}

textarea:focus {
    border: 1px solid #6366f1 !important;
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.25) !important;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 17px;
    font-weight: 600;
    margin-top: 15px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.35);
}

.result-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
}

.spam {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
}

.not-spam {
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.4);
}

.result-icon {
    font-size: 42px;
}

.result-title {
    font-size: 22px;
    font-weight: 500;
    color: white;
    margin-top: 8px;
}

.result-text {
    color: #94a3b8;
    font-size: 15px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# TEXT PREPROCESSING
# ==================================================

ps = PorterStemmer()


def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric words
    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ==================================================
# LOAD MODEL
# ==================================================

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))


# ==================================================
# HEADER
# ==================================================

st.markdown("""<div class="header">
<div class="header-icon">🛡️</div>
<div class="header-title">SMS Spam Classifier</div>
<div class="header-subtitle">Intelligent SMS Spam Detection</div>
</div>""", unsafe_allow_html=True)


# ==================================================
# INPUT
# ==================================================

st.markdown(
    '<div class="input-title">Analyze your message</div>',
    unsafe_allow_html=True
)


input_sms = st.text_area(
    "Enter your message",
    placeholder="Enter or paste your SMS here...",
    height=180,
    label_visibility="collapsed"
)


# ==================================================
# PREDICTION
# ==================================================

if st.button("🔍  Analyze Message"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")

    else:

        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result

        if result == 0:

            st.markdown("""<div class="result-box not-spam">
<div class="result-icon">🟢</div>
<div class="result-title">NOT SPAM</div>
<div class="result-text">This message appears to be legitimate.</div>
</div>""", unsafe_allow_html=True)

        else:

            st.markdown("""<div class="result-box spam">
<div class="result-icon">🔴</div>
<div class="result-title">SPAM</div>
<div class="result-text">This message appears to be suspicious.</div>
</div>""", unsafe_allow_html=True)