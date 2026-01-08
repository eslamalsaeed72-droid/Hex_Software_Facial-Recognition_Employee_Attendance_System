
import streamlit as st
from datetime import datetime, timedelta
import cv2
import numpy as np
import face_recognition
import sqlite3
import pickle
import pandas as pd
import os
from PIL import Image
import base64
from pathlib import Path

# ============ DATABASE CLASS ============
class AttendanceDatabase:
    def __init__(self, db_name='attendance.db'):
        self.db_name = db_name
        self.connection = None
        self.initialize_database()
    
    def initialize_database(self):
        try:
            self.connection = sqlite3.connect(self.db_name)
            cursor = self.connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    employee_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT,
                    department TEXT,
                    face_encoding BLOB,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_id INTEGER NOT NULL,
                    check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    check_out_time TIMESTAMP,
                    date DATE,
                    FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
                )
            """)
            
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
    
    def add_employee(self, name, email, phone, department, face_encoding):
        try:
            cursor = self.connection.cursor()
            face_encoding_blob = pickle.dumps(face_encoding)
            
            cursor.execute("""
                INSERT INTO employees (name, email, phone, department, face_encoding)
                VALUES (?, ?, ?, ?, ?)
            """, (name, email, phone, department, face_encoding_blob))
            
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"Email {email} already exists")
            return False
        except Exception as e:
            print(f"Error adding employee: {e}")
            return False
    
    def get_all_employees(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT employee_id, name, email, phone, department FROM employees")
            return cursor.fetchall()
        except Exception as e:
            print(f"Error retrieving employees: {e}")
            return []
    
    def get_face_encodings(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT employee_id, name, face_encoding FROM employees")
            results = cursor.fetchall()
            
            encodings = []
            employee_ids = []
            names = []
            
            for employee_id, name, face_encoding_blob in results:
                encoding = pickle.loads(face_encoding_blob)
                encodings.append(encoding)
                employee_ids.append(employee_id)
                names.append(name)
            
            return encodings, employee_ids, names
        except Exception as e:
            print(f"Error retrieving face encodings: {e}")
            return [], [], []
    
    def mark_attendance(self, employee_id):
        try:
            cursor = self.connection.cursor()
            current_date = datetime.now().date()
            
            cursor.execute("""
                SELECT attendance_id FROM attendance
                WHERE employee_id = ? AND date = ? AND check_out_time IS NULL
            """, (employee_id, current_date))
            
            if cursor.fetchone():
                return False
            
            cursor.execute("""
                INSERT INTO attendance (employee_id, date)
                VALUES (?, ?)
            """, (employee_id, current_date))
            
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_report(self, start_date=None, end_date=None):
        try:
            cursor = self.connection.cursor()
            
            if start_date and end_date:
                cursor.execute("""
                    SELECT e.name, e.email, a.date, a.check_in_time
                    FROM attendance a
                    JOIN employees e ON a.employee_id = e.employee_id
                    WHERE a.date BETWEEN ? AND ?
                    ORDER BY a.date DESC, a.check_in_time DESC
                """, (start_date, end_date))
            else:
                cursor.execute("""
                    SELECT e.name, e.email, a.date, a.check_in_time
                    FROM attendance a
                    JOIN employees e ON a.employee_id = e.employee_id
                    ORDER BY a.date DESC, a.check_in_time DESC
                """)
            
            results = cursor.fetchall()
            df = pd.DataFrame(results, columns=["Employee Name", "Email", "Date", "Check-in Time"])
            return df
        except Exception as e:
            print(f"Error generating report: {e}")
            return pd.DataFrame()

# ============ IMAGE PROCESSOR CLASS ============
class ImageProcessor:
    @staticmethod
    def load_image_from_upload(uploaded_file):
        try:
            image_pil = Image.open(uploaded_file)
            image_rgb = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return None
    
    @staticmethod
    def convert_bgr_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============ FACIAL RECOGNITION CLASS ============
class FacialRecognitionEngine:
    def __init__(self):
        prototxt_path = '/content/models/deploy.prototxt'
        model_path = '/content/models/res10_300x300_ssd_iter_140000.caffemodel'
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model files not found!")
        
        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except Exception as e:
            raise Exception(f"Error loading Caffe model: {e}")
    
    def detect_and_encode_faces(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            return face_locations, face_encodings
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return [], []
    
    def recognize_faces(self, image, known_encodings, known_ids, known_names, tolerance=0.6):
        try:
            face_locations, face_encodings = self.detect_and_encode_faces(image)
            
            recognized_names = []
            recognized_ids = []
            distances_list = []
            
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                
                if min_distance <= tolerance:
                    recognized_names.append(known_names[min_distance_index])
                    recognized_ids.append(known_ids[min_distance_index])
                    distances_list.append(min_distance)
                else:
                    recognized_names.append("Unknown")
                    recognized_ids.append(None)
                    distances_list.append(min_distance)
            
            return face_locations, recognized_names, recognized_ids, distances_list
        except Exception as e:
            print(f"Error recognizing faces: {e}")
            return [], [], [], []
    
    def draw_recognition_results(self, image, face_locations, names, distances):
        try:
            annotated_image = image.copy()
            
            for (top, right, bottom, left), name, distance in zip(face_locations, names, distances):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 2)
                
                confidence = (1 - distance) * 100
                label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else "Unknown"
                
                cv2.rectangle(annotated_image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(annotated_image, label, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            return annotated_image
        except Exception as e:
            print(f"Error drawing results: {e}")
            return image

# ============ INITIALIZE GLOBALS ============
db = AttendanceDatabase('attendance.db')
try:
    fr_engine = FacialRecognitionEngine()
except Exception as e:
    st.error(f"Error loading facial recognition engine: {e}")
    fr_engine = None

# ============ STREAMLIT APP ============
st.set_page_config(
    page_title="Facial Recognition Attendance System",
    page_icon="📷",
    layout="wide"
)

st.markdown("""
    <style>
        h1 { color: #0066cc; text-align: center; }
        h2 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.title("👤 Facial Recognition Employee Attendance System")
st.markdown("---")

if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'marked_attendance' not in st.session_state:
    st.session_state.marked_attendance = []

st.sidebar.title("🔧 Navigation Menu")
app_mode = st.sidebar.radio(
    "Select Module",
    ["Home", "Employee Registration", "Mark Attendance", "View Reports", "Employee Management"]
)

if app_mode == "Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        total_employees = len(db.get_all_employees())
        st.metric(label="Total Employees", value=total_employees)
    with col2:
        today_attendance = len(st.session_state.marked_attendance)
        st.metric(label="Today's Check-ins", value=today_attendance)
    with col3:
        st.metric(label="System Status", value="✅ Active")
    
    st.markdown("---")
    st.subheader("📋 About This System")
    st.write("✅ Real-time facial recognition\n✅ Automatic attendance logging\n✅ Comprehensive reporting")

elif app_mode == "Employee Registration":
    st.subheader("📝 Register New Employee")
    
    col1, col2 = st.columns(2)
    with col1:
        employee_name = st.text_input("Employee Full Name")
        employee_email = st.text_input("Email Address")
        employee_phone = st.text_input("Phone Number")
        employee_dept = st.selectbox("Department", ["HR", "IT", "Sales", "Finance", "Operations"])
    
    with col2:
        uploaded_file = st.file_uploader("Upload Employee Photo", type=["jpg", "jpeg", "png"])
        employee_image = None
        if uploaded_file:
            employee_image = ImageProcessor.load_image_from_upload(uploaded_file)
            if employee_image is not None:
                st.image(ImageProcessor.convert_bgr_to_rgb(employee_image), caption="Uploaded Photo")
    
    if st.button("🔐 Register Employee"):
        if not employee_name or not employee_email:
            st.error("Please fill in all required fields")
        elif employee_image is None:
            st.error("Please provide an employee photo")
        elif fr_engine is None:
            st.error("Facial recognition engine not available")
        else:
            face_locations, face_encodings = fr_engine.detect_and_encode_faces(employee_image)
            
            if len(face_encodings) == 0:
                st.error("No face detected!")
            elif len(face_encodings) > 1:
                st.error("Multiple faces detected!")
            else:
                success = db.add_employee(employee_name, employee_email, employee_phone, employee_dept, face_encodings[0])
                if success:
                    st.success(f"✅ Employee {employee_name} registered!")
                    st.balloons()
                else:
                    st.error("Failed to register. Email may exist.")

elif app_mode == "Mark Attendance":
    st.subheader("✅ Mark Attendance")
    
    known_encodings, known_ids, known_names = db.get_face_encodings()
    
    if len(known_encodings) == 0:
        st.warning("No registered employees!")
    else:
        uploaded_file = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            test_image = ImageProcessor.load_image_from_upload(uploaded_file)
            
            if test_image is not None and fr_engine is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📸 Photo")
                    st.image(ImageProcessor.convert_bgr_to_rgb(test_image))
                
                face_locations, recognized_names, recognized_ids, distances = fr_engine.recognize_faces(
                    test_image, known_encodings, known_ids, known_names
                )
                
                with col2:
                    st.subheader("🔍 Results")
                    if len(face_locations) == 0:
                        st.warning("No faces detected!")
                    else:
                        for name, emp_id, distance in zip(recognized_names, recognized_ids, distances):
                            if name != "Unknown":
                                st.write(f"**{name}** - {(1-distance)*100:.1f}%")
                                if st.button("✅ Check In", key=f"checkin_{emp_id}"):
                                    if db.mark_attendance(emp_id):
                                        st.success(f"Marked for {name}!")
                                    else:
                                        st.warning("Already checked in!")
                            else:
                                st.error("Unknown!")
                
                annotated = fr_engine.draw_recognition_results(test_image, face_locations, recognized_names, distances)
                st.subheader("🎯 Visualization")
                st.image(ImageProcessor.convert_bgr_to_rgb(annotated))

elif app_mode == "View Reports":
    st.subheader("📊 Reports")
    
    report_type = st.radio("Type", ["Daily", "Weekly", "Custom"])
    
    if report_type == "Daily":
        today = datetime.now().date()
        df = db.get_attendance_report(str(today), str(today))
        st.dataframe(df) if len(df) > 0 else st.info("No records")
    
    elif report_type == "Weekly":
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        df = db.get_attendance_report(str(week_start), str(today))
        st.dataframe(df) if len(df) > 0 else st.info("No records")
    
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start")
        with col2:
            end_date = st.date_input("End")
        
        if st.button("Generate"):
            df = db.get_attendance_report(str(start_date), str(end_date))
            if len(df) > 0:
                st.dataframe(df)
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, f"report_{start_date}.csv", "text/csv")
            else:
                st.info("No records")

elif app_mode == "Employee Management":
    st.subheader("👥 Employees")
    
    employees = db.get_all_employees()
    
    if len(employees) == 0:
        st.info("No employees")
    else:
        data = [{"ID": e[0], "Name": e[1], "Email": e[2], "Phone": e[3], "Dept": e[4]} for e in employees]
        st.dataframe(pd.DataFrame(data))
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(employees))
        with col2:
            depts = len(set([e[4] for e in employees]))
            st.metric("Departments", depts)
        with col3:
            st.metric("Status", "✅ Active")

st.markdown("---")
st.markdown("<div style='text-align: center;'><p>🚀 v1.0</p></div>", unsafe_allow_html=True)
