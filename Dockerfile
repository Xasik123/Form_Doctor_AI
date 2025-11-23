# 1. Берем базовый образ Python 3.10 (он легкий и стабильный)
FROM python:3.10-slim

# 2. Устанавливаем системные библиотеки (для OpenCV и FFmpeg)
# Это то, что вы мучились устанавливать на Windows, здесь ставится одной командой.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3. Создаем рабочую папку внутри контейнера
WORKDIR /app

# 4. Копируем список библиотек (создадим его следующим шагом)
COPY requirements.txt .

# 5. Устанавливаем библиотеки Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Копируем весь ваш код в контейнер
COPY . .

# 7. Команда запуска (та самая, которую вы вводили в консоль)
# Мы меняем host на 0.0.0.0, чтобы сервер был виден из интернета
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "10000"]