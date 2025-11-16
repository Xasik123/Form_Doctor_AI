# Dockerfile
# Используем базовый образ, который уже имеет Python 3.10
FROM python:3.10-slim

# Установка необходимых системных зависимостей для OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY . /app
WORKDIR /app

# Установка Python-зависимостей из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Установка Streamlit
RUN pip install streamlit

# Команда для запуска Streamlit
CMD streamlit run app.py
