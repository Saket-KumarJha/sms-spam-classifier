
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==================================================
# NLTK RESOURCES
# ==================================================

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
# CUSTOM CSS - COMPACT DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e1b4b);
}

.block-container {
    max-width: 600px;
    padding-top: 20px;
    padding-bottom: 20px;
    padding-left: 20px;
    padding-right: 20px;
}

.header {
    text-align: center;
    margin-bottom: 18px;
}

.header-icon {
    font-size: 30px;
    margin-bottom: 3px;
    margin-top : 40px;
}

.header-title {
    font-size: 27px;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
}

.header-subtitle {
    font-size: 13px;
    color: #94a3b8;
}

.input-title {
    font-size: 16px;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 5px;
}

/* Compact text area */
textarea {
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #475569 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}

textarea:focus {
    border: 1px solid #6366f1 !important;
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.25) !important;
}

/* Compact analyze button */
.stButton > button {
    width: 100%;
    min-height: 40px;
    border-radius: 9px;
    border: none;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 14px;
    font-weight: 600;
    margin-top: 5px;
    padding: 6px 12px;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 12px rgba(99, 102, 241, 0.3);
}

/* Compact prediction result */
.result-box {
    margin-top: 16px;
    padding: 15px;
    border-radius: 12px;
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
    font-size: 28px;
}

.result-title {
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-top: 4px;
}

.result-text {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 4px;
}

/* Mobile-friendly layout */
@media (max-width: 600px) {
    .block-container {
        padding-top: 15px;
        padding-left: 15px;
        padding-right: 15px;
    }

    .header-title {
        font-size: 23px;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TEXT PREPROCESSING
# ==================================================

ps = PorterStemmer()

# Cache stopwords instead of loading them repeatedly
english_stopwords = set(stopwords.words("english"))


def transform_text(text):
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]

    # Remove stopwords and punctuation
    tokens = [
        word for word in tokens
        if word not in english_stopwords
        and word not in string.punctuation
    ]

    # Apply stemming
    tokens = [ps.stem(word) for word in tokens]

    return " ".join(tokens)

# ==================================================
# LOAD MODEL AND VECTORIZER
# ==================================================

@st.cache_resource
def load_model():
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    with open("model.pkl", "rb") as file:
        classifier = pickle.load(file)

    return vectorizer, classifier


tfidf, model = load_model()

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="header">
    <div class="header-icon">🛡️</div>
    <div class="header-title">SMS Spam Classifier</div>
    <div class="header-subtitle">
        Intelligent SMS Spam Detection
    </div>
</div>
""", unsafe_allow_html=True)

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
    height=110,
    label_visibility="collapsed"
)

# ==================================================
# PREDICTION
# ==================================================

if st.button("🔍  Analyze Message"):

    if not input_sms.strip():
        st.warning("Please enter a message first.")

    else:
        try:
            # 1. Preprocess message
            transformed_sms = transform_text(input_sms)

            # 2. Convert text into TF-IDF features
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict spam or ham
            result = model.predict(vector_input)[0]

            # 4. Display result
            if result == 0:
                st.markdown("""
                <div class="result-box not-spam">
                    <div class="result-icon">🟢</div>
                    <div class="result-title">NOT SPAM</div>
                    <div class="result-text">
                        This message appears to be legitimate.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="result-box spam">
                    <div class="result-icon">🔴</div>
                    <div class="result-title">SPAM</div>
                    <div class="result-text">
                        This message appears to be suspicious.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except LookupError:
            st.error(
                "Required NLTK resources are missing. "
                "Please check the app logs and ensure "
                "punkt_tab, punkt, and stopwords are downloaded."
            )

        except Exception as e:
            st.error(
                "An error occurred while analyzing the message. "
                "Please check the app logs."
            )
