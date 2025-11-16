#!/bin/bash
# Обновление системных пакетов
sudo apt-get update -y

# Установка MediaPipe (самостоятельно установит OpenCV)
/home/adminuser/venv/bin/pip install mediapipe==0.10.21 --no-cache-dir

# Установка остальных пакетов
/home/adminuser/venv/bin/pip install -r requirements.txt
