FROM python:3.10-slim

WORKDIR /app

COPY code/deployment/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/app/streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]