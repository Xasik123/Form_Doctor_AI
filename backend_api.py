# backend_api.py - ИСПРАВЛЕННАЯ ЛОГИКА РАСПОЗНАВАНИЯ (ПРИСЕДАНИЯ)

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import Optional
import shutil
import sys
import os
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import base64
from google import genai
from google.genai import types
import ffmpeg 
import traceback 
import hashlib 
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# --- 1. НАСТРОЙКА БАЗЫ ДАННЫХ ---
# Умная настройка пути:
if sys.platform.startswith('win'):
    # Если Windows (ваш компьютер) -> сохраняем в папке проекта
    DATABASE_URL = "sqlite:///./sql_app.db"
else:
    # Если Linux (Hugging Face / Docker) -> сохраняем в папке /tmp (там всегда есть права)
    # Обратите внимание на 4 слэша: sqlite:////tmp/...
    DATABASE_URL = "sqlite:////tmp/sql_app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=True)
    language_code = Column(String, default="ru")

class AnalysisHistory(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    analysis_date = Column(DateTime, default=datetime.utcnow)
    final_score = Column(Integer)
    report_text = Column(String)
    
Base.metadata.create_all(bind=engine)

# --- 2. СХЕМЫ PYDANTIC ---
class UserCreate(BaseModel):
    email: str
    password: str 
    language_code: Optional[str] = "ru"

class UserInDB(BaseModel):
    id: int
    email: str
    language_code: str

# --- 3. ИНИЦИАЛИЗАЦИЯ ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
GEMINI_API_KEY = "AIzaSyCi8ygU26S8xSeHENw2mzQjCZk0_lCDQgw" 
app = FastAPI(title="Form Doctor AI Backend")

# --- 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_password_hash(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def convert_to_mp4(input_path, output_path):
    try:
        (ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', preset='medium', pix_fmt='yuv420p', vf='scale=1280:-2').overwrite_output().run(capture_stdout=True, capture_stderr=True))
        return True
    except: return False

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def analyze_frame_results(image):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        return results, image

def draw_landmarks_on_frame(image):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        return image

# Функции углов
def get_knee_angle(lm):
    try: return calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
    except: return None
def get_torso_back_angle(lm):
    try: return calculate_angle([lm[12].x, lm[12].y], [lm[24].x, lm[24].y], [lm[26].x, lm[26].y])
    except: return None

# ИСПРАВЛЕННАЯ ЛОГИКА РАСПОЗНАВАНИЯ
def recognize_exercise(angles_history, torso_back_angles_history):
    if not angles_history or not torso_back_angles_history: return "Неизвестное упражнение"
    
    min_knee = min(angles_history)
    min_torso = min(torso_back_angles_history)

    # 1. Приседания (Исправлено)
    # Если колено согнуто сильно (< 90 градусов), мы считаем это приседом, 
    # игнорируя наклон корпуса (так как приседы бывают разные)
    if min_knee < 90: 
        return "Приседания"
    
    # 2. Становая тяга
    # Если ноги прямые (угол > 130), но корпус сильно наклонен (< 140)
    elif min_knee > 130 and min_torso < 140: 
        return "Становая тяга"
    
    # 3. Жимы стоя/сидя
    elif min_torso > 165: 
        return "Армейский жим" if np.mean(angles_history) > 140 else "Тяга верхнего блока"
    
    # 4. Изоляция
    elif min_knee > 170 and min_torso > 170: 
        return "Сгибание рук со штангой"
        
    return "Неизвестное упражнение"

# --- 5. ГЛАВНАЯ ЛОГИКА АНАЛИЗА ---
def analyze_video(video_path, api_key): 
    cap = cv2.VideoCapture(video_path)
    angles_history = []; torso_back_angles_history = []; error_frames = {}; frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 5 != 0: frame_count += 1; continue

        results, image = analyze_frame_results(frame)
        try:
            lm = results.pose_landmarks.landmark
            if get_knee_angle(lm): angles_history.append(get_knee_angle(lm))
            if get_torso_back_angle(lm): torso_back_angles_history.append(get_torso_back_angle(lm))
            
            if get_torso_back_angle(lm) and get_torso_back_angle(lm) < 165 and 'Back_Round' not in error_frames:
                 error_frames['Back_Round'] = (frame.copy(), get_torso_back_angle(lm))
        except: pass
        frame_count += 1
    cap.release()

    if not angles_history: return None, None, None, "Не удалось обнаружить человека."

    deepest_angle = min(angles_history)
    exercise = recognize_exercise(angles_history, torso_back_angles_history) 

    try:
        client = genai.Client(api_key=api_key)
        error_image_base64 = None
        if 'Back_Round' in error_frames:
            annotated_frame = draw_landmarks_on_frame(error_frames['Back_Round'][0])
            _, buffer = cv2.imencode('.jpg', annotated_frame) 
            error_image_base64 = base64.b64encode(buffer).decode('utf-8')

        focus_error = "Округление поясницы/прогиб" if 'Back_Round' in error_frames else "Отсутствует"
        
        prompt = f"""
        Действуй как профессиональный тренер.
        Я определил упражнение как: {exercise} (по углам суставов).
        Минимальный угол колена: {deepest_angle:.2f} градусов.
        Возможная ошибка: {focus_error}.
        
        Твоя задача:
        1. Подтверди, похоже ли это на {exercise} или скорректируй название.
        2. Дай детальный разбор техники.
        3. Если есть ошибка, дай упражнение для исправления.
        """
        
        contents = [types.Part.from_text(text=prompt)]
        if error_image_base64:
             contents.append(types.Part.from_bytes(data=base64.b64decode(error_image_base64), mime_type='image/jpeg'))
             
        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
        return deepest_angle, error_image_base64, response.text
    except Exception as e:
        return None, None, f"Ошибка API: {e}"

# --- 6. ЭНДПОИНТЫ API ---

@app.post("/register", response_model=UserInDB)
def register_new_user(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, email=user.email):
        raise HTTPException(status_code=400, detail="Email занят.")
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password, language_code=user.language_code)
    db.add(db_user); db.commit(); db.refresh(db_user)
    return db_user

@app.post("/analyze_form")
async def analyze_form_endpoint(video_file: UploadFile = File(...), db: Session = Depends(get_db)):
    temp_input = ""; temp_output = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(video_file.file, tmp)
            temp_input = tmp.name
        temp_output = temp_input + "_converted.mp4"

        if not convert_to_mp4(temp_input, temp_output):
            raise HTTPException(status_code=400, detail="Ошибка конвертации видео.")
        
        deepest_angle, error_image_base64, report = analyze_video(temp_output, GEMINI_API_KEY)
        
        if "Ошибка API" in report:
            raise HTTPException(status_code=500, detail=report)

        return {"status": "success", "exercise_detected": "Автоматически", "min_angle": f"{deepest_angle:.2f}°", "report": report}

    except Exception as e:
        import traceback
        if os.path.exists(temp_input): os.unlink(temp_input)
        if os.path.exists(temp_output): os.unlink(temp_output)
        raise HTTPException(status_code=500, detail=f"CRITICAL ERROR:\n{traceback.format_exc()}")
