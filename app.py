import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email Spam Detection App")
st.write("Enter an email message and check whether it is *Spam* or *Not Spam (Ham)*.")

# -------------------------------
# Dataset
# -------------------------------
data = {
    "text": [
        "You won a lottery!",
        "Claim prize by clicking!",
        "Click on amount to redeem reward!",
        "Limited offer for you account!",
        "Addhar OTP",
        "MOdule Assessment",
        "Your ticket is confirmed with IRCTC",
        "Registration number for SSC Exams.",
        "Credit card offer for low interest.",
        "Win cash prize!",
        "ES Class is scheduled on monday"
    ],
    "label": ["spam","spam","spam","spam","ham","ham","ham","ham","spam","spam","ham"]
}

df = pd.DataFrame(data)

# -------------------------------
# Train Model
# -------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('nb', MultinomialNB(alpha=0.1))
])

model.fit(X_train, y_train)

# -------------------------------
# User Input
# -------------------------------
email_text = st.text_area(
    "✉️ Enter Email Text Below:",
    height=150,
    placeholder="Type or paste email content here..."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Check Spam"):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        prediction = model.predict([email_text])[0]
        probability = model.predict_proba([email_text])[0]

        spam_index = list(model.classes_).index("spam")
        ham_index = list(model.classes_).index("ham")

        if prediction == "spam":
            st.error("🚨 This is a SPAM Email")
            st.write("Spam Probability:", round(probability[spam_index], 2))
        else:
            st.success("✅ This is NOT a Spam Email")
            st.write("Ham Probability:", round(probability[ham_index], 2))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using NLP, TF-IDF & Naive Bayes")