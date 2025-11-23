# Используем Python 3.10
FROM python:3.10-slim

# Создаем пользователя (требование безопасности Hugging Face)
RUN useradd -m -u 1000 user
WORKDIR /app

# Устанавливаем системные библиотеки для видео (от имени root)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY --chown=user . .

# Переключаемся на обычного пользователя
USER user

# Открываем порт 7860 (ОБЯЗАТЕЛЬНО ДЛЯ HF)
EXPOSE 7860

# Запускаем сервер на порту 7860
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "7860"]

