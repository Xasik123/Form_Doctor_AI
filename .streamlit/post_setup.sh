#!/bin/bash
# Обновление системных пакетов
sudo apt-get update -y
sudo apt-get install -y ffmpeg libsm6 libxext6 # Установка системных зависимостей

# Установка MediaPipe и OpenCV в два этапа для обхода конфликтов
/home/adminuser/venv/bin/pip install mediapipe==0.10.21
/home/adminuser/venv/bin/pip install opencv-python-headless==4.11.0.86 --force-reinstall
/home/adminuser/venv/bin/pip install -r requirements.txt
