import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# ---------------------------------------------------------
# Page Configuration (Must be first Streamlit call)
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector (Linear SVM)", page_icon="📩", layout="centered"
)


# ---------------------------------------------------------
# 1. Model Training & Caching
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    """Loads dataset from URL, vectorizes using TF-IDF,

    and trains a LinearSVC model.
    """
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, _, y_train, _ = train_test_split(
        df["message"],
        df["label_num"],
        test_size=0.2,
        random_state=42,
        stratify=df["label_num"],
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LinearSVC(random_state=42)
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


# Load model and vectorizer
with st.spinner("Training Linear SVM Model..."):
    vectorizer, model = load_and_train_model()

# ---------------------------------------------------------
# 2. UI Header & Description
# ---------------------------------------------------------
st.title("📩 SMS & Email Spam Classifier")
st.markdown("""
This web application uses a **Linear Support Vector Machine (LinearSVC)** combined with **TF-IDF Vectorization** to classify incoming messages as **Legitimate (Ham)** or **Spam**.
""")

st.divider()

# ---------------------------------------------------------
# 3. User Input Section
# ---------------------------------------------------------
st.subheader("Enter a Message to Test:")

user_input = st.text_area(
    "Message Text:", height=120, placeholder="Type or paste SMS message here..."
)

# Predict Button
if st.button("🔍 Analyze Message", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        # Transform input text
        input_tfidf = vectorizer.transform([user_input])

        # Prediction & Decision Function Score
        prediction = model.predict(input_tfidf)[0]
        decision_score = model.decision_function(input_tfidf)[0]

        st.divider()

        # Display Results
        if prediction == 1:
            st.error("🚨 **RESULT: SPAM DETECTED**")
        else:
            st.success("✅ **RESULT: LEGITIMATE MESSAGE (HAM)**")

        # Display Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Label", "SPAM" if prediction == 1 else "HAM")
        with col2:
            st.metric("Decision Margin Score", f"{decision_score:.3f}")

        st.caption(
            "Note: Decision Margin Score > 0 indicates Spam, while < 0 indicates Ham. "
            "Larger absolute values signify higher confidence."
        )

        # Highlight Matched Terms
        st.subheader("Detected Feature Breakdown")
        feature_names = vectorizer.get_feature_names_out()
        nonzero_indices = input_tfidf.nonzero()[1]

        matched_words = [feature_names[i] for i in nonzero_indices]

        if matched_words:
            st.write("Keywords recognized by vectorizer:")
            st.write(", ".join([f"`{word}`" for word in matched_words]))
        else:
            st.info(
                "No learned vocabulary keywords were found in this message."
            )

# Footer
st.divider()
st.caption("Powered by Mera")
