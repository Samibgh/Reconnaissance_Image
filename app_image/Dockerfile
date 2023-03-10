FROM python:3.11-slim

COPY requirements.txt /app/requirements.txt

WORKDIR /app 
RUN pip install -r requirements.txt

COPY . /

COPY app.py app.py

CMD ["streamlit","run", "app.py"]