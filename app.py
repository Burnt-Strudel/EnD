import streamlit as st
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("API классификации обращений")

st.write("Введите текст обращения пользователя:")

text = st.text_area("Текст обращения")

if st.button("Определить категорию"):

    if text.strip() == "":
        st.warning("Введите текст обращения")
    else:
        text_vectorized = vectorizer.transform([text])

        prediction = model.predict(text_vectorized)[0]

        st.success(f"Категория обращения: {prediction}")