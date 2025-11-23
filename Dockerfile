# Используем официальный образ Python 3.10 (slim версия для уменьшения размера)
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
# Добавляем --fix-missing и --no-install-recommends для стабильности
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем requirements.txt (для кэширования слоев Docker)
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем остальной код проекта
COPY . .

# Открываем порт (Render ожидает 10000 по умолчанию, но uvicorn настроим ниже)
EXPOSE 10000

# Команда запуска
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "10000"]
