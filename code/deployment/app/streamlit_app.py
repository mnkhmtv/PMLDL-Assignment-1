import streamlit as st
import requests

st.set_page_config(page_title="Прогноз успеха поста", page_icon="📝")

st.title("Прогноз успеха поста в Telegram-канале")

with st.form("prediction_form"):
    text = st.text_area("Текст поста", "", height=150)
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Час публикации (0–23)", 0, 23, 12)
    with col2:
        weekday = st.selectbox("День недели (0 — понедельник, 6 — воскресенье)", range(7),
                            format_func=lambda x: ["Пн","Вт","Ср","Чт","Пт","Сб","Вс"][x])
    submit = st.form_submit_button("Получить прогноз")

if submit:
    if not text.strip():
        st.error("Пожалуйста, введите текст поста.")
    else:
        params = {
            "text": text,
            "hour": hour,
            "weekday": weekday
        }
        url = "http://api:8000/predict"

        with st.spinner("Выполняется прогноз..."):
            try:
                response = requests.post(url, json=params)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Прогноз просмотров: **{data['predicted_views']}**")
                    st.info(f"Прогноз общего количества реакций: **{data['predicted_total_reactions']}**")
                else:
                    st.error(f"Ошибка сервера API: {response.status_code}")
            except Exception as e:
                st.error(f"Не удалось связаться с API. Проверьте, что он запущен.\n\nОшибка: {e}")

st.markdown("""
---
<small>
Innopolis University, 2025
Practical Machine Learning and Deep Learning
</small>
""", unsafe_allow_html=True)