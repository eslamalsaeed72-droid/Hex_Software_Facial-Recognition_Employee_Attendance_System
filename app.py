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
            self.connection = sqlite3.connect(self.db_name, check_same_thread=False)
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
            st.error(f"Database error: {e}")
    
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
            st.warning(f"Email {email} already exists")
            return False
        except Exception as e:
            st.error(f"Error adding employee: {e}")
            return False
    
    def get_all_employees(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT employee_id, name, email, phone, department FROM employees")
            return cursor.fetchall()
        except Exception as e:
            st.error(f"Error retrieving employees: {e}")
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
            st.error(f"Error retrieving face encodings: {e}")
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
            st.error(f"Error marking attendance: {e}")
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
            st.error(f"Error generating report: {e}")
            return pd.DataFrame()

# ============ IMAGE PROCESSOR CLASS ============
class ImageProcessor:
    @staticmethod
    def load_image_from_upload(uploaded_file):
        try:
            image_pil = Image.open(uploaded_file)
            image_rgb = np.array(image_pil)
            # Fix: Check if image has alpha channel (RGBA) and convert
            if image_rgb.shape[-1] == 4:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    
    @staticmethod
    def convert_bgr_to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ============ FACIAL RECOGNITION CLASS ============
class FacialRecognitionEngine:
    def __init__(self):
        # --- FIXED PATHS FOR GITHUB/STREAMLIT ---
        # Get the directory where the current script (app.py) is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to the models folder inside the repo
        prototxt_path = os.path.join(current_dir, "models", "deploy.prototxt")
        model_path = os.path.join(current_dir, "models", "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Debugging: Print paths to logs (visible in Streamlit Cloud logs)
        print(f"Loading models from: {prototxt_path}")
        
        if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
            st.error(f"⚠️ Model files not found at: {prototxt_path}")
            st.stop()
        
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
            st.error(f"Error detecting faces: {e}")
            return [], []
    
    def recognize_faces(self, image, known_encodings, known_ids, known_names, tolerance=0.5):
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
            st.error(f"Error recognizing faces: {e}")
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
            st.error(f"Error drawing results: {e}")
            return image

# ============ INITIALIZE GLOBALS ============
db = AttendanceDatabase('attendance.db')

# Safe initialization of Recognition Engine
try:
    fr_engine = FacialRecognitionEngine()
except Exception as e:
    st.error(f"Error initializing engine: {e}")
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
        .stButton button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.title("👤 Facial Recognition Employee Attendance System")
st.markdown("---")

if 'marked_attendance' not in st.session_state:
    st.session_state.marked_attendance = []

# Navigation
st.sidebar.title("🔧 Navigation Menu")
app_mode = st.sidebar.radio(
    "Select Module",
    ["Home", "Employee Registration", "Mark Attendance", "View Reports", "Employee Management"]
)

# --- HOME MODULE ---
if app_mode == "Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        total_employees = len(db.get_all_employees())
        st.metric(label="Total Employees", value=total_employees)
    with col2:
        # Simple count of today's records
        today_df = db.get_attendance_report(str(datetime.now().date()), str(datetime.now().date()))
        st.metric(label="Today's Check-ins", value=len(today_df))
    with col3:
        st.metric(label="System Status", value="✅ Active")
    
    st.markdown("---")
    st.subheader("📋 About This System")
    st.info("Welcome to the Attendance System. Use the sidebar to navigate.")
    st.write("✅ Real-time facial recognition")
    st.write("✅ Automatic attendance logging")
    st.write("✅ Comprehensive reporting (CSV Export)")

# --- REGISTRATION MODULE ---
elif app_mode == "Employee Registration":
    st.subheader("📝 Register New Employee")
    
    col1, col2 = st.columns(2)
    with col1:
        employee_name = st.text_input("Employee Full Name")
        employee_email = st.text_input("Email Address")
        employee_phone = st.text_input("Phone Number")
        employee_dept = st.selectbox("Department", ["HR", "IT", "Sales", "Finance", "Operations", "Engineering"])
    
    with col2:
        uploaded_file = st.file_uploader("Upload Employee Photo", type=["jpg", "jpeg", "png"])
        employee_image = None
        if uploaded_file:
            employee_image = ImageProcessor.load_image_from_upload(uploaded_file)
            if employee_image is not None:
                st.image(ImageProcessor.convert_bgr_to_rgb(employee_image), caption="Preview", width=250)
    
    if st.button("🔐 Register Employee"):
        if not employee_name or not employee_email:
            st.error("Please fill in all required fields")
        elif employee_image is None:
            st.error("Please provide an employee photo")
        elif fr_engine is None:
            st.error("Facial recognition engine not available")
        else:
            with st.spinner("Processing..."):
                face_locations, face_encodings = fr_engine.detect_and_encode_faces(employee_image)
            
            if len(face_encodings) == 0:
                st.error("No face detected! Please try a clearer photo.")
            elif len(face_encodings) > 1:
                st.error("Multiple faces detected! Please ensure only one person is in the photo.")
            else:
                success = db.add_employee(employee_name, employee_email, employee_phone, employee_dept, face_encodings[0])
                if success:
                    st.success(f"✅ Employee {employee_name} registered successfully!")
                    st.balloons()
                else:
                    st.error("Failed to register. Email might already exist.")

# --- ATTENDANCE MODULE ---
elif app_mode == "Mark Attendance":
    st.subheader("✅ Mark Attendance")
    
    known_encodings, known_ids, known_names = db.get_face_encodings()
    
    if len(known_encodings) == 0:
        st.warning("No registered employees! Please register employees first.")
    else:
        uploaded_file = st.file_uploader("Upload Photo to Check In", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            test_image = ImageProcessor.load_image_from_upload(uploaded_file)
            
            if test_image is not None and fr_engine is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(ImageProcessor.convert_bgr_to_rgb(test_image), caption="Input Image")
                
                with st.spinner("Analyzing..."):
                    face_locations, recognized_names, recognized_ids, distances = fr_engine.recognize_faces(
                        test_image, known_encodings, known_ids, known_names
                    )
                
                with col2:
                    st.subheader("🔍 Results")
                    if len(face_locations) == 0:
                        st.warning("No faces detected in the image.")
                    else:
                        for name, emp_id, distance in zip(recognized_names, recognized_ids, distances):
                            if name != "Unknown":
                                confidence = (1 - distance) * 100
                                st.success(f"Identified: **{name}** ({confidence:.1f}%)")
                                
                                if st.button(f"Confirm Check-In for {name}", key=f"btn_{emp_id}"):
                                    if db.mark_attendance(emp_id):
                                        st.success(f"✅ Attendance marked for {name} at {datetime.now().strftime('%H:%M:%S')}")
                                    else:
                                        st.warning(f"⚠️ {name} has already checked in today.")
                            else:
                                st.error("Unknown Person Detected")
                
                # Show visualized image
                annotated = fr_engine.draw_recognition_results(test_image, face_locations, recognized_names, distances)
                st.image(ImageProcessor.convert_bgr_to_rgb(annotated), caption="Analyzed Output")

# --- REPORTS MODULE ---
elif app_mode == "View Reports":
    st.subheader("📊 Attendance Reports")
    
    report_type = st.radio("Select Report Type", ["Daily", "Weekly", "Custom Date Range"], horizontal=True)
    
    df = pd.DataFrame()
    today = datetime.now().date()
    
    if report_type == "Daily":
        st.caption(f"Showing records for: {today}")
        df = db.get_attendance_report(str(today), str(today))
        
    elif report_type == "Weekly":
        week_start = today - timedelta(days=today.weekday())
        st.caption(f"Showing records from {week_start} to {today}")
        df = db.get_attendance_report(str(week_start), str(today))
        
    else:
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=today - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", value=today)
        
        if st.button("Generate Report"):
            df = db.get_attendance_report(str(start_date), str(end_date))
    
    # Display Data
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, f"attendance_report_{datetime.now().date()}.csv", "text/csv")
    else:
        st.info("No attendance records found for the selected period.")

# --- MANAGEMENT MODULE ---
elif app_mode == "Employee Management":
    st.subheader("👥 Employee Database")
    
    employees = db.get_all_employees()
    
    if len(employees) == 0:
        st.info("No employees registered yet.")
    else:
        data = [{"ID": e[0], "Name": e[1], "Email": e[2], "Phone": e[3], "Department": e[4]} for e in employees]
        df_emps = pd.DataFrame(data)
        st.dataframe(df_emps, use_container_width=True)
        
        st.markdown("### Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Employees", len(employees))
        with col2:
            st.metric("Departments", len(df_emps['Department'].unique()))

st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Developed by Hex Software | v1.0</div>", unsafe_allow_html=True)
