import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Movie Review Analyzer", page_icon="🎬", layout="wide")


# ---------------- LIGHT BACKGROUND ----------------
st.markdown("""
<style>

.stApp{
background-image: url("https://images.unsplash.com/photo-1517602302552-471fe67acf66");
background-size: cover;
background-position: center;
background-attachment: fixed;
}

/* LIGHT OVERLAY FOR READABILITY */
.block-container{
background: rgba(255,255,255,0.88);
padding:30px;
border-radius:10px;
}

/* TITLE */
.title{
font-size:70px;
font-weight:900;
text-align:center;
color:#0a1f44;
}

/* SUBTITLE */
.subtitle{
font-size:26px;
text-align:center;
color:#333;
margin-bottom:30px;
}

/* CARDS */
.card{
background:white;
padding:20px;
border-radius:12px;
box-shadow:0 4px 10px rgba(0,0,0,0.1);
margin-bottom:20px;
}

/* RESULT BOX */
.result{
background:#eef4ff;
padding:20px;
border-radius:10px;
font-size:20px;
font-weight:600;
}

textarea{
font-size:18px !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ----------------
st.markdown('<div class="title">🎬 AI Movie Review Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Movie Sentiment & Rating using Machine Learning</div>', unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
df = pd.read_csv("movie_prediction.csv")
df = df.dropna().drop_duplicates()

X = df["review"]
y = df["sentiment"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# ---------------- TFIDF ----------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=2000,C=2,solver="liblinear")
model.fit(X_train_tfidf,y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test,y_pred)


# ---------------- DASHBOARD INFO ----------------
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Dataset Size")
    st.write(df.shape)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Model Accuracy")
    st.write(round(accuracy*100,2),"%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Algorithm")
    st.write("Logistic Regression")
    st.markdown('</div>', unsafe_allow_html=True)



# ---------------- REVIEW INPUT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("✍️ Enter Movie Review")

review = st.text_area("Write your movie review")

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- FUNCTIONS ----------------
def predict_sentiment(review):

    vector = tfidf.transform([review])
    return model.predict(vector)[0]


def predict_star_rating(review):

    vector = tfidf.transform([review])

    prob = model.predict_proba(vector)[0][1]

    if prob < 0.2:
        stars = 1
    elif prob < 0.4:
        stars = 2
    elif prob < 0.6:
        stars = 3
    elif prob < 0.8:
        stars = 4
    else:
        stars = 5

    return stars,prob


# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Review"):

    if review.strip()=="":
        st.warning("Please enter a review")

    else:

        sentiment = predict_sentiment(review)
        stars,prob = predict_star_rating(review)

        st.markdown('<div class="result">', unsafe_allow_html=True)

        st.write("🎭 Sentiment:",sentiment)
        st.write("⭐ Rating:", "⭐"*stars)
        st.write("📊 Confidence:", round(prob*100,2),"%")

        # -------- SENTIMENT GAUGE --------
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob*100,
            title = {'text': "Sentiment Confidence"},
            gauge = {'axis': {'range': [0,100]}}
        ))

        st.plotly_chart(fig)

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- WORD CLOUD + WORDS ----------------
col1,col2 = st.columns(2)

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("☁️ WordCloud")

    text = " ".join(df["review"])

    wordcloud = WordCloud(width=400,height=200,background_color="white").generate(text)

    fig,ax = plt.subplots(figsize=(4,2))

    ax.imshow(wordcloud)
    ax.axis("off")

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🔎 Important Words")

    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]

    top_positive = np.argsort(coefficients)[-10:]
    top_negative = np.argsort(coefficients)[:10]

    # Create two columns for side-by-side display
    pos_col, neg_col = st.columns(2)

    with pos_col:
        st.write("### 👍 Positive Words")
        for i in top_positive:
            st.write(feature_names[i])

    with neg_col:
        st.write("### 👎 Negative Words")
        for i in top_negative:
            st.write(feature_names[i])

    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")
st.write("🎬 AI Movie Review Analyzer | Streamlit + Machine Learning")