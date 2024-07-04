import subprocess
import importlib.util

def check_install(package):
    try:
        importlib.import_module(package)
        print(f"{package} is already installed")
    except ModuleNotFoundError:
        print(f"{package} is not installed, attempting to install...")
        try:
            subprocess.check_call(['pip', 'install', 'opencv-python'])
            print(f"{package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")

# Check and install OpenCV
check_install('cv2')





import cv2
# Load Haar Cascade classifiers
face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to detect faces and eyes using Haar Cascades
def detect_faces_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast for better detection
    
    # Detect faces
    faces = face_cascade_default.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        face_roi = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame, 'Eye', (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Function to capture and process video
def capture_video():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces and eyes using Haar Cascades
        frame = detect_faces_eyes(frame)
        
        cv2.imshow('Advanced Haar Cascade Face Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

capture_video()
