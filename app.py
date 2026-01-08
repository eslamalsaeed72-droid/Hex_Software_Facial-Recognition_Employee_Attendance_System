# Facial Recognition Employee Attendance System
# Streamlit Web Application - Production Ready
# Compatible with: Local, Colab, Streamlit Cloud

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
import hashlib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Facial Recognition Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
    <style>
        .main {
            padding: 0rem 1rem;
        }
        
        h1 {
            color: #0066cc;
            text-align: center;
            padding: 20px 0;
            border-bottom: 3px solid #0066cc;
        }
        
        h2 {
            color: #0066cc;
            border-bottom: 2px solid #0066cc;
            padding-bottom: 10px;
            margin-top: 20px;
        }
        
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .danger-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE CLASS
# ============================================================

class AttendanceDatabase:
    """Handle all database operations for employee and attendance management."""
    
    def __init__(self, db_name='attendance.db'):
        self.db_name = db_name
        self.connection = None
        self.initialize_database()
    
    def initialize_database(self):
        """Create database tables if they don't exist."""
        try:
            self.connection = sqlite3.connect(self.db_name)
            cursor = self.connection.cursor()
            
            # Employees table
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
            
            # Attendance table
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
        """Add a new employee to the database."""
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
            st.error(f"❌ Email '{email}' already exists in the system!")
            return False
        except Exception as e:
            st.error(f"Error adding employee: {e}")
            return False
    
    def get_all_employees(self):
        """Retrieve all employees from the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT employee_id, name, email, phone, department FROM employees")
            return cursor.fetchall()
        except Exception as e:
            st.error(f"Error retrieving employees: {e}")
            return []
    
    def get_face_encodings(self):
        """Retrieve all face encodings for recognition."""
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
        """Record attendance for an employee."""
        try:
            cursor = self.connection.cursor()
            current_date = datetime.now().date()
            
            cursor.execute("""
                SELECT attendance_id FROM attendance
                WHERE employee_id = ? AND date = ? AND check_out_time IS NULL
            """, (employee_id, current_date))
            
            if cursor.fetchone():
                return False  # Already checked in
            
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
        """Generate attendance report for specified date range."""
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
    
    def delete_all_data(self):
        """Delete all employees and attendance records from the database."""
        try:
            cursor = self.connection.cursor()
            cursor.execute("DELETE FROM attendance")
            cursor.execute("DELETE FROM employees")
            self.connection.commit()
            return True
        except Exception as e:
            st.error(f"Error deleting data: {e}")
            return False
    
    def close_connection(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

# ============================================================
# IMAGE PROCESSOR CLASS
# ============================================================

class ImageProcessor:
    """Handle all image processing operations."""
    
    @staticmethod
    def load_image_from_upload(uploaded_file):
        """Load image from Streamlit uploaded file."""
        try:
            image_pil = Image.open(uploaded_file)
            image_rgb = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return image_bgr
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    
    @staticmethod
    def convert_bgr_to_rgb(image):
        """Convert image from BGR to RGB color space."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
        """Resize image maintaining aspect ratio."""
        try:
            h, w = image.shape[:2]
            
            if width is None and height is None:
                return image
            
            if width is None:
                ratio = height / float(h)
                width = int(w * ratio)
            elif height is None:
                ratio = width / float(w)
                height = int(h * ratio)
            
            resized = cv2.resize(image, (width, height), interpolation=inter)
            return resized
        except Exception as e:
            st.error(f"Error resizing image: {e}")
            return image

# ============================================================
# FACIAL RECOGNITION ENGINE
# ============================================================

class FacialRecognitionEngine:
    """Handle facial detection and recognition operations."""
    
    def __init__(self):
        """Initialize facial recognition engine."""
        try:
            # Try to find models in different locations
            possible_paths = [
                './models/deploy.prototxt',
                '/content/models/deploy.prototxt',
                'models/deploy.prototxt',
            ]
            
            prototxt_path = None
            model_path = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    prototxt_path = path
                    model_path = path.replace('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
                    break
            
            if not prototxt_path or not os.path.exists(model_path):
                st.warning("⚠️ Caffe models not found. Using fallback face detection.")
                self.net = None
            else:
                self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load Caffe model: {e}. Using fallback detection.")
            self.net = None
    
    def detect_and_encode_faces(self, image):
        """Detect all faces in an image and generate their encodings."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use face_recognition library for reliable detection
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            return face_locations, face_encodings
        except Exception as e:
            st.error(f"Error detecting faces: {e}")
            return [], []
    
    def recognize_faces(self, image, known_encodings, known_ids, known_names, tolerance=0.6):
        """Recognize faces in an image against database of known faces."""
        try:
            if len(known_encodings) == 0:
                st.warning("No registered employees to compare against.")
                return [], [], [], []
            
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
        """Draw rectangles and labels on image showing recognition results."""
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

# ============================================================
# INITIALIZE GLOBALS
# ============================================================

@st.cache_resource
def init_db():
    """Initialize database connection (cached)."""
    return AttendanceDatabase('attendance.db')

@st.cache_resource
def init_fr_engine():
    """Initialize facial recognition engine (cached)."""
    return FacialRecognitionEngine()

db = init_db()
fr_engine = init_fr_engine()

# Initialize session state
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'marked_attendance' not in st.session_state:
    st.session_state.marked_attendance = []

# ============================================================
# MAIN APPLICATION
# ============================================================

# Title and Navigation
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.title("👤 Facial Recognition Employee Attendance System")

with col3:
    # Delete button in top right
    if st.button("🗑️ Admin", key="admin_btn"):
        st.session_state.show_admin = True

st.markdown("---")

st.sidebar.title("🔧 Navigation")
app_mode = st.sidebar.radio(
    "Select Module",
    ["🏠 Home", "👤 Register Employee", "✅ Mark Attendance", "📊 View Reports", "👥 Manage Employees"]
)

# ============================================================
# HOME PAGE
# ============================================================

if app_mode == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_employees = len(db.get_all_employees())
        st.metric(label="👥 Total Employees", value=total_employees)
    
    with col2:
        today_attendance = len(st.session_state.marked_attendance)
        st.metric(label="✅ Today's Check-ins", value=today_attendance)
    
    with col3:
        st.metric(label="⚙️ System Status", value="Active ✓")
    
    st.markdown("---")
    
    st.subheader("📋 About This System")
    st.info("""
    **Facial Recognition Employee Attendance System v1.0**
    
    A modern, AI-powered solution for automated employee attendance tracking.
    
    ### ✨ Key Features:
    - 🔍 Real-time facial recognition using deep learning
    - 📱 Easy employee registration with photo upload
    - ⏱️ Automatic attendance logging with timestamps
    - 📊 Comprehensive attendance reports (daily, weekly, custom)
    - 💾 SQLite database for reliable data storage
    - 🎨 User-friendly Streamlit web interface
    - 📈 Employee statistics and analytics
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 Getting Started")
        st.write("""
        1. **Register Employees** - Upload clear photos
        2. **Mark Attendance** - Upload photo to check in
        3. **View Reports** - Track attendance history
        4. **Manage Data** - View all employees
        """)
    
    with col2:
        st.subheader("💡 Tips")
        st.write("""
        - Use clear, well-lit photos
        - Ensure face is fully visible
        - Only one person per photo
        - Similar photos work better
        """)

# ============================================================
# EMPLOYEE REGISTRATION PAGE
# ============================================================

elif app_mode == "👤 Register Employee":
    st.subheader("📝 Register New Employee")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Employee Information")
        employee_name = st.text_input("👤 Full Name", placeholder="e.g., John Doe")
        employee_email = st.text_input("📧 Email Address", placeholder="e.g., john@company.com")
        employee_phone = st.text_input("📱 Phone Number", placeholder="e.g., +20 1234567890")
        employee_dept = st.selectbox("🏢 Department", ["HR", "IT", "Sales", "Finance", "Operations", "Other"])
    
    with col2:
        st.write("### Upload Photo")
        uploaded_file = st.file_uploader("📸 Select employee photo", type=["jpg", "jpeg", "png"])
        employee_image = None
        
        if uploaded_file:
            employee_image = ImageProcessor.load_image_from_upload(uploaded_file)
            if employee_image is not None:
                st.image(ImageProcessor.convert_bgr_to_rgb(employee_image), caption="✅ Photo loaded", use_column_width=True)
    
    st.markdown("---")
    
    if st.button("🔐 Register Employee", key="register_btn", use_container_width=True):
        if not employee_name or not employee_email:
            st.error("❌ Please fill in all required fields (Name & Email)")
        elif employee_image is None:
            st.error("❌ Please upload an employee photo")
        else:
            with st.spinner("🔍 Processing face..."):
                face_locations, face_encodings = fr_engine.detect_and_encode_faces(employee_image)
                
                if len(face_encodings) == 0:
                    st.error("❌ No face detected! Please upload a clear photo where the face is visible.")
                elif len(face_encodings) > 1:
                    st.error("❌ Multiple faces detected! Please upload a photo with only one person.")
                else:
                    success = db.add_employee(
                        employee_name,
                        employee_email,
                        employee_phone,
                        employee_dept,
                        face_encodings[0]
                    )
                    
                    if success:
                        st.success(f"✅ Employee '{employee_name}' registered successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Registration failed. Email may already exist.")

# ============================================================
# MARK ATTENDANCE PAGE
# ============================================================

elif app_mode == "✅ Mark Attendance":
    st.subheader("✅ Mark Employee Attendance")
    
    known_encodings, known_ids, known_names = db.get_face_encodings()
    
    if len(known_encodings) == 0:
        st.warning("⚠️ No registered employees found. Please register employees first.")
    else:
        st.write(f"📊 **System Ready** - {len(known_encodings)} employees in database")
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("📸 Upload photo for attendance marking", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            test_image = ImageProcessor.load_image_from_upload(uploaded_file)
            
            if test_image is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📸 Uploaded Photo")
                    st.image(ImageProcessor.convert_bgr_to_rgb(test_image), use_column_width=True)
                
                with st.spinner("🔍 Analyzing faces..."):
                    face_locations, recognized_names, recognized_ids, distances = fr_engine.recognize_faces(
                        test_image,
                        known_encodings,
                        known_ids,
                        known_names
                    )
                
                with col2:
                    st.subheader("🔍 Recognition Results")
                    
                    if len(face_locations) == 0:
                        st.warning("⚠️ No faces detected in the image")
                    else:
                        for name, emp_id, distance in zip(recognized_names, recognized_ids, distances):
                            confidence = (1 - distance) * 100
                            
                            if name != "Unknown":
                                col_a, col_b = st.columns([2, 1])
                                with col_a:
                                    st.write(f"**{name}**")
                                    st.caption(f"Confidence: {confidence:.1f}%")
                                with col_b:
                                    if st.button("✅ Check In", key=f"checkin_{emp_id}"):
                                        if db.mark_attendance(emp_id):
                                            st.success(f"✅ Marked for {name}!")
                                            st.session_state.marked_attendance.append({
                                                'name': name,
                                                'time': datetime.now()
                                            })
                                        else:
                                            st.warning(f"⚠️ {name} already checked in today")
                            else:
                                st.error(f"❌ Unknown person - Confidence: {confidence:.1f}%")
                
                st.markdown("---")
                st.subheader("🎯 Recognition Visualization")
                
                annotated = fr_engine.draw_recognition_results(
                    test_image,
                    face_locations,
                    recognized_names,
                    distances
                )
                st.image(ImageProcessor.convert_bgr_to_rgb(annotated), use_column_width=True)

# ============================================================
# VIEW REPORTS PAGE
# ============================================================

elif app_mode == "📊 View Reports":
    st.subheader("📊 Attendance Reports")
    
    report_type = st.radio("📋 Select Report Type", ["Daily", "Weekly", "Custom Date Range"], horizontal=True)
    
    if report_type == "Daily":
        today = datetime.now().date()
        attendance_df = db.get_attendance_report(str(today), str(today))
        
        st.write(f"**📅 Daily Report - {today}**")
        
        if len(attendance_df) == 0:
            st.info("ℹ️ No attendance records for today")
        else:
            st.dataframe(attendance_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Check-ins", len(attendance_df))
            with col2:
                unique_employees = attendance_df['Employee Name'].nunique()
                st.metric("Unique Employees", unique_employees)
    
    elif report_type == "Weekly":
        today = datetime.now().date()
        week_start = today - timedelta(days=today.weekday())
        attendance_df = db.get_attendance_report(str(week_start), str(today))
        
        st.write(f"**📅 Weekly Report - {week_start} to {today}**")
        
        if len(attendance_df) == 0:
            st.info("ℹ️ No attendance records for this week")
        else:
            st.dataframe(attendance_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Check-ins", len(attendance_df))
            with col2:
                unique_employees = attendance_df['Employee Name'].nunique()
                st.metric("Unique Employees", unique_employees)
    
    else:  # Custom Date Range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("📅 Start Date")
        with col2:
            end_date = st.date_input("📅 End Date")
        
        if st.button("📈 Generate Report", use_container_width=True):
            if start_date > end_date:
                st.error("❌ Start date must be before end date")
            else:
                attendance_df = db.get_attendance_report(str(start_date), str(end_date))
                
                if len(attendance_df) == 0:
                    st.info("ℹ️ No attendance records found for the selected date range")
                else:
                    st.write(f"**📊 Attendance Report: {start_date} to {end_date}**")
                    st.dataframe(attendance_df, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Check-ins", len(attendance_df))
                    with col2:
                        unique_employees = attendance_df['Employee Name'].nunique()
                        st.metric("Unique Employees", unique_employees)
                    with col3:
                        days = (end_date - start_date).days + 1
                        st.metric("Days Covered", days)
                    
                    # Download button
                    csv = attendance_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Report as CSV",
                        data=csv,
                        file_name=f"attendance_report_{start_date}_{end_date}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

# ============================================================
# EMPLOYEE MANAGEMENT PAGE
# ============================================================

elif app_mode == "👥 Manage Employees":
    st.subheader("👥 Employee Management")
    
    employees = db.get_all_employees()
    
    if len(employees) == 0:
        st.info("ℹ️ No employees registered in the system yet")
    else:
        # Display employees in a table
        employee_data = []
        for emp_id, name, email, phone, dept in employees:
            employee_data.append({
                'ID': emp_id,
                'Name': name,
                'Email': email,
                'Phone': phone,
                'Department': dept
            })
        
        df_employees = pd.DataFrame(employee_data)
        
        st.write(f"### 📋 All Employees ({len(employees)})")
        st.dataframe(df_employees, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("📈 Employee Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Total Employees", len(employees))
        
        with col2:
            departments = [emp[4] for emp in employees]
            unique_depts = len(set(departments))
            st.metric("🏢 Departments", unique_depts)
        
        with col3:
            st.metric("⚙️ System Status", "✓ Active")
        
        # Department distribution
        st.subheader("🏢 Department Distribution")
        dept_counts = df_employees['Department'].value_counts()
        st.bar_chart(dept_counts)

# ============================================================
# ADMIN PANEL - DELETE DATA
# ============================================================

if st.session_state.get("show_admin", False):
    st.markdown("---")
    st.warning("⚠️ **ADMIN PANEL - DELETE ALL DATA**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔐 Password Protection")
        admin_password = st.text_input("Enter Admin Password:", type="password")
    
    with col2:
        st.write("### ⚠️ Confirmation")
        confirm = st.checkbox("I understand this will delete ALL employees and attendance records")
    
    if st.button("🗑️ DELETE ALL DATA", key="delete_all_btn"):
        # Hash the password (simple security)
        password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
        correct_hash = hashlib.sha256("HexSoftware2026".encode()).hexdigest()  # Default password
        
        if password_hash == correct_hash and confirm:
            if db.delete_all_data():
                st.success("✅ All data has been deleted successfully!")
                st.session_state.show_admin = False
                st.rerun()
            else:
                st.error("❌ Error deleting data!")
        elif not confirm:
            st.error("❌ Please confirm the deletion")
        else:
            st.error("❌ Incorrect password!")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: #888; font-size: 12px;">
        <p>🚀 <strong>Facial Recognition Employee Attendance System v1.0</strong></p>
        <p>Built with Python | OpenCV | Streamlit | Deep Learning</p>
        <p>Developed by <strong>Hex Software</strong> • Made with ❤️ for automated attendance tracking</p>
        <p>© 2026 Eslam Alsaeed</p>
    </div>
""", unsafe_allow_html=True)
