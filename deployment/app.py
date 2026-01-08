import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="LDA Topic Modeling - Roblox Reviews",
    layout="wide"
)

st.title("Topic Modeling Roblox Reviews (LDA)")

# =========================
# TOPIC LABELS
# =========================
topic_labels = {
    0: "Love & Rewards Satisfaction",
    1: "Social Experience & Community",
    2: "Performance & Technical Issues",
    3: "System, Updates & Moderation Issues",
    4: "General Enjoyment & Popularity"
}

# =========================
# NLP SETUP
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

# =========================
# LOAD RESOURCE
# =========================
import os

@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.dirname(__file__))

    with open(os.path.join(base_dir, "dataset", "roblox_dictionary.pkl"), "rb") as f:
        dictionary = pickle.load(f)

    with open(os.path.join(base_dir, "dataset", "roblox_corpus.pkl"), "rb") as f:
        corpus = pickle.load(f)

    lda_model = gensim.models.LdaModel.load(
        os.path.join(base_dir, "dataset", "lda_bow_model")
    )

    return dictionary, corpus, lda_model



# =========================
# REVIEW TO TOPIC
# =========================
st.markdown("---")
st.subheader("Analisis Topik Review Baru")

user_input = st.text_area("Masukkan Review Roblox")

if st.button("Analisis Topik"):
    if user_input.strip() == "":
        st.warning("Review tidak boleh kosong")
    else:
        tokens = preprocess(user_input)
        bow = dictionary.doc2bow(tokens)

        topic_probs = lda_model.get_document_topics(bow)

        best_topic, best_prob = max(topic_probs, key=lambda x: x[1])
        best_topic_display = best_topic + 1
        label = topic_labels.get(best_topic, "Unknown Topic")

        st.success(f"Review ini termasuk ke **{label} (Topik {best_topic_display})**")


        topic_words = lda_model.show_topic(best_topic, topn=10)
        topic_keywords = [w for w, _ in topic_words]

        st.markdown("**Kata kunci topik ini:**")
        st.write(", ".join(topic_keywords))
