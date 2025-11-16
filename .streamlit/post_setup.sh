#!/bin/bash
# Обновление системных пакетов
sudo apt-get update -y

# Установка OpenCV и MediaPipe без учета зависимостей
# Используем прямой путь к pip в среде Streamlit Cloud
/home/adminuser/venv/bin/pip install mediapipe==0.10.21 --no-deps
/home/adminuser/venv/bin/pip install opencv-python==4.11.0.86 --no-deps

# Установка остальных пакетов
/home/adminuser/venv/bin/pip install -r requirements.txt
