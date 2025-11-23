# backend_api.py - ПОЛНАЯ ВЕРСИЯ (AUTH + HISTORY + AI)

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from typing import Optional, List
import shutil
import os
import sys
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import base64
from google import genai
from google.genai import types
import ffmpeg 
import traceback 
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timedelta
from jose import JWTError, jwt

# --- 1. КОНФИГУРАЦИЯ ---
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_CHANGE_THIS" # В реальном проекте спрятать в Secrets!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

if sys.platform.startswith('win'):
    DATABASE_URL = "sqlite:///./sql_app.db"
else:
    DATABASE_URL = "sqlite:////tmp/sql_app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- 2. МОДЕЛИ ДАННЫХ ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    language_code = Column(String, default="ru")
    
    # Связь с историей
    history_items = relationship("AnalysisHistory", back_populates="owner")

class AnalysisHistory(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise = Column(String)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    min_angle = Column(Float)
    report_text = Column(String)
    
    owner = relationship("User", back_populates="history_items")
    
Base.metadata.create_all(bind=engine)

# --- 3. СХЕМЫ PYDANTIC ---
class UserCreate(BaseModel):
    email: str
    password: str 
    language_code: Optional[str] = "ru"

class UserResponse(BaseModel):
    id: int
    email: str
    class Config: from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class HistoryItem(BaseModel):
    exercise: str
    min_angle: float
    report_text: str
    analysis_date: datetime
    class Config: from_attributes = True

# --- 4. ИНСТРУМЕНТЫ ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCi8ygU26S8xSeHENw2mzQjCZk0_lCDQgw") 
app = FastAPI(title="Form Doctor AI Backend")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- 5. ФУНКЦИИ ---

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None: raise credentials_exception
    return user

# ... (ЗДЕСЬ ДОЛЖНЫ БЫТЬ ФУНКЦИИ АНАЛИЗА: calculate_angle, convert_to_mp4, analyze_video) ...
# ... (Я скопировал их ниже для полноты, но в редакторе убедитесь, что они есть) ...

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

def get_knee_angle(lm):
    try: return calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
    except: return None
def get_torso_back_angle(lm):
    try: return calculate_angle([lm[12].x, lm[12].y], [lm[24].x, lm[24].y], [lm[26].x, lm[26].y])
    except: return None

def recognize_exercise(angles_history, torso_back_angles_history):
    if not angles_history or not torso_back_angles_history: return "Неизвестное упражнение"
    min_knee = min(angles_history); min_torso = min(torso_back_angles_history)
    if min_knee < 90: return "Приседания"
    elif min_knee > 130 and min_torso < 140: return "Становая тяга"
    return "Неизвестное упражнение"

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
        prompt = f"Упражнение: {exercise}. Мин. угол: {deepest_angle}. Ошибка: {focus_error}. Дай совет."
        
        contents = [types.Part.from_text(text=prompt)]
        if error_image_base64:
             contents.append(types.Part.from_bytes(data=base64.b64decode(error_image_base64), mime_type='image/jpeg'))
             
        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents)
        
        # Возвращаем и название упражнения для истории!
        return deepest_angle, error_image_base64, response.text, exercise 
    except Exception as e:
        return None, None, f"Ошибка API: {e}", "Ошибка"

# --- 6. ЭНДПОИНТЫ ---

@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email уже существует")
    import hashlib
    hashed_password = hashlib.sha256(user.password.encode()).hexdigest()
    new_user = User(email=user.email, hashed_password=hashed_password, language_code=user.language_code)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    import hashlib
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user: raise HTTPException(status_code=400, detail="Неверный email или пароль")
    
    # Проверка пароля
    hashed_input = hashlib.sha256(form_data.password.encode()).hexdigest()
    if hashed_input != user.hashed_password:
        raise HTTPException(status_code=400, detail="Неверный email или пароль")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/analyze_form")
async def analyze_form_endpoint(
    video_file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user), # <-- ТРЕБУЕТСЯ ВХОД
    db: Session = Depends(get_db)
):
    temp_input = ""; temp_output = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            shutil.copyfileobj(video_file.file, tmp)
            temp_input = tmp.name
        temp_output = temp_input + ".mp4"

        if not convert_to_mp4(temp_input, temp_output):
            temp_output = temp_input # Пробуем оригинал, если конвертация не удалась
        
        # Анализ
        deepest_angle, error_image_base64, report, exercise_name = analyze_video(temp_output, GEMINI_API_KEY)
        
        if "Ошибка API" in report: raise HTTPException(status_code=500, detail=report)
        
        # --- СОХРАНЕНИЕ В ИСТОРИЮ ---
        history_item = AnalysisHistory(
            user_id=current_user.id,
            exercise=exercise_name,
            min_angle=deepest_angle,
            report_text=report[:500] # Сохраняем начало отчета (или весь, если база позволяет)
        )
        db.add(history_item)
        db.commit()

        return {
            "status": "success", 
            "exercise": exercise_name,
            "min_angle": f"{deepest_angle:.2f}", 
            "error_image_base64": error_image_base64, 
            "report": report,
            "history_saved": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
    finally:
        if os.path.exists(temp_input): os.unlink(temp_input)
        if os.path.exists(temp_output) and temp_output != temp_input: os.unlink(temp_output)

@app.get("/history", response_model=List[HistoryItem])
def read_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(AnalysisHistory).filter(AnalysisHistory.user_id == current_user.id).all()

