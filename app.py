# app.py - ПОЛНЫЙ ИНТЕГРИРОВАННЫЙ КОД
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from google import genai
from google.genai import types

# Инициализация моделей
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ФУНКЦИЯ: Расчет Угла
def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ФУНКЦИЯ: Анализ Кадра
def analyze_frame_results(image):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        return results

# ФУНКЦИЯ: Угол Колена
def get_knee_angle(landmarks):
    try:
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        return calculate_angle(hip, knee, ankle)
    except:
        return None

# ФУНКЦИЯ: Проверка Вальгуса
def check_valgus_valgus(landmarks):
    try:
        r_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
        r_knee_x = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
        l_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
        l_knee_x = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x
        
        is_r_valgus = r_knee_x < r_hip_x 
        is_l_valgus = l_knee_x > l_hip_x 
        
        if is_r_valgus or is_l_valgus:
            return True, "Односторонний вальгус"
        return False, None
    except:
        return False, None

# ФУНКЦИЯ: Угол Корпуса/Таза (для Butt Wink)
def get_torso_hip_angle(landmarks):
    try:
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        return calculate_angle(shoulder, hip, knee)
    except:
        return None

# ФУНКЦИЯ: Анализ Видео и Генерация Отчета
def analyze_video(video_path, api_key):
    cap = cv2.VideoCapture(video_path)

    angles_history = []
    torso_hip_angles_history = [] 
    valgus_count = 0
    frame_count = 0
    butt_wink_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = analyze_frame_results(frame)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            knee_angle = get_knee_angle(landmarks)
            if knee_angle is not None:
                angles_history.append(knee_angle)

            torso_hip_angle = get_torso_hip_angle(landmarks)
            if torso_hip_angle is not None:
                torso_hip_angles_history.append(torso_hip_angle)

            is_valgus, valgus_type = check_valgus_valgus(landmarks)
            if is_valgus:
                valgus_count += 1
                
        except:
            pass
        frame_count += 1

    cap.release()

    if frame_count == 0 or not angles_history:
        return None, None, None, "Не удалось обнаружить человека или прочитать видео."

    # --- СИНТЕЗ РЕЗУЛЬТАТОВ И ЛОГИКА BUTT WINK ---
    deepest_angle = min(angles_history)
    valgus_percentage = (valgus_count / frame_count) * 100
    
    if len(torso_hip_angles_history) > 1:
        min_knee_angle_idx = np.argmin(angles_history)
        if min_knee_angle_idx > 0 and min_knee_angle_idx < len(torso_hip_angles_history):
            start_torso_angle = torso_hip_angles_history[0] 
            deepest_torso_angle = torso_hip_angles_history[min_knee_angle_idx] 
            
            if (deepest_torso_angle - start_torso_angle) < -10: 
                butt_wink_detected = True

    # --- ГЕНЕРАЦИЯ ОТЧЕТА GEMINI ---
    try:
        client = genai.Client(api_key=api_key)
        
        focus_error = ""
        focus_recommendation = ""

        if butt_wink_detected:
            focus_error = "Округление поясницы ('Butt Wink')"
            focus_recommendation = "Для устранения Butt Wink и повышения мобильности голеностопа добавьте в разминку упражнение: Wall Ankle Mobilization (Мобилизация голеностопа у стены). Выполняйте 3 подхода по 10 повторений на каждую ногу."
        elif valgus_percentage > 20:
            focus_error = f"Вальгусный завал коленей ({valgus_percentage:.1f}%)"
            focus_recommendation = "Для немедленного устранения вальгусного завала и укрепления ягодичных мышц добавьте в разминку упражнение: Banded Clamshells (Ракушки с резинкой). Выполняйте 3 подхода по 15 повторений на каждую ногу."
        else:
            focus_error = "Отсутствует"
            focus_recommendation = "Продолжайте фокусироваться на контроле в нижней точке и равномерном подъеме."

        prompt = f"""
        Ты — ИИ-тренер 'Form Doctor'. Твоя задача — дать обратную связь.
        - Глубина (мин. угол колена): {deepest_angle:.2f} градусов.
        - Фокус ошибки: {focus_error}.
        
        ПРАВИЛА: 
        1. Хвали за глубину (меньше 90 градусов - отлично).
        2. Главный фокус отчета - на {focus_error}, если это не 'Отсутствует'.
        3. Рекомендация: {focus_recommendation}.
        
        Сгенерируй мотивирующий отчет с заголовками.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return deepest_angle, butt_wink_detected, valgus_percentage, response.text
    
    except Exception as e:
        return None, None, None, f"Ошибка API или генерации: {e}"


# --- 3. ГЛАВНАЯ ФУНКЦИЯ STREAMLIT (ИНТЕРФЕЙС) ---

# --- 3. ГЛАВНАЯ ФУНКЦИЯ STREAMLIT (ИНТЕРФЕЙС) ---

def main():
    st.set_page_config(layout="centered")
    st.title("🏋️ Form Doctor AI: Анализ Техники Приседаний")
    st.markdown("Загрузите видео (MP4/MOV) для получения экспертного отчета. **Рекомендуется съемка сбоку!**")
    st.divider()
    
    # Получение API ключа
    api_key = st.text_input("🔑 Введите ваш Gemini API Key (обязательно):", type="password")
    
    # Загрузка видео
    uploaded_file = st.file_uploader("🎥 Загрузите файл MP4 или MOV:", type=["mp4", "mov"])
    
    # --- КНОПКА ЗАПУСКА И ОСНОВНАЯ ЛОГИКА ---
    # Кнопка всегда видна. Анализ запустится, только если условия выполнены.
    
    if st.button("🚀 Запустить Анализ Формы"):
        
        # 1. Проверка условий ПЕРЕД запуском
        if uploaded_file is None:
            st.error("Пожалуйста, сначала загрузите видео.")
            return # Выход из функции
        if not api_key:
            st.error("Пожалуйста, сначала введите API Key.")
            return # Выход из функции
        
        # 2. Если все условия выполнены:
        
        # Временное сохранение видео для обработки OpenCV
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        with st.spinner("Анализ видео... Пожалуйста, подождите (до 30 секунд)."):
            deepest_angle, butt_wink, valgus_pct, report = analyze_video(video_path, api_key)
        
        # 3. Вывод результатов
        if deepest_angle is not None and "Ошибка API" not in report:
            st.success("✅ Анализ Завершен!")
            
            col1, col2 = st.columns(2)
            col1.metric("Мин. Угол Колена (Глубина)", f"{deepest_angle:.2f}°", help="Меньше 90° - ниже параллели.")
            col2.metric("Butt Wink (Округление Таза)", "Обнаружено 🚩" if butt_wink else "Не обнаружено ✅")
            
            if valgus_pct > 20 and not butt_wink:
                 st.warning(f"⚠️ Вальгусный завал замечен в {valgus_pct:.1f}% кадров. См. рекомендации.")

            st.subheader("📝 Отчет ИИ-Тренера:")
            st.markdown(report)
        
        else:
            st.error(report)

# --- ЗАПУСК ПРИЛОЖЕНИЯ ---
if __name__ == '__main__':
    main()
