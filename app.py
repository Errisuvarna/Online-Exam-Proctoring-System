from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
import base64
import numpy as np
import cv2
import torch
import time
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
ALLOWED_OBJECTS = ["person"]

# Ensure necessary folders exist
if not os.path.exists("reports"):
    os.makedirs("reports")
if not os.path.exists("violations"):
    os.makedirs("violations")

# Database connection helper
def get_db_connection():
    conn = sqlite3.connect("exam_system.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]
        conn = get_db_connection()
        conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                     (username, password, role))
        conn.commit()
        conn.close()
        return redirect("/login")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session["user_id"] = user["id"]
            session["role"] = user["role"]
            if user["role"] == "admin":
                return redirect("/admin")
            else:
                return redirect("/exam")
    return render_template("login.html")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if session.get("role") != "admin":
        return redirect("/")
    conn = get_db_connection()
    cursor = conn.cursor()
    if request.method == "POST":
        question = request.form["question"]
        options = [request.form["option1"], request.form["option2"],
                   request.form["option3"], request.form["option4"]]
        correct_answer = request.form["correct_answer"]
        cursor.execute(
            "INSERT INTO questions (question, option1, option2, option3, option4, correct_answer) "
            "VALUES (?, ?, ?, ?, ?, ?)", (question, *options, correct_answer))
        conn.commit()
    cursor.execute("SELECT * FROM questions")
    questions = cursor.fetchall()
    conn.close()
    return render_template("admin_dashboard.html", questions=questions)

@app.route("/exam", methods=["GET"])
def exam():
    if "user_id" not in session or session.get("role") != "student":
        return redirect("/")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM questions")
    questions = cursor.fetchall()
    conn.close()
    return render_template("exam.html", questions=questions)

@app.route("/submit_exam", methods=["POST"])
def submit_exam():
    if "user_id" not in session or session.get("role") != "student":
        return redirect("/")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM questions")
    questions = cursor.fetchall()
    score = 0
    for question in questions:
        user_answer = request.form.get(f"q{question['id']}")
        correct_index = int(question["correct_answer"])
        correct_option = question[f"option{correct_index}"]
        if user_answer == correct_option:
            score += 1
    conn.close()
    return render_template("result.html", score=score, total=len(questions))

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/detect_cheating", methods=["POST"])
def detect_cheating():
    data = request.get_json()
    user_id = data.get("user_id", "unknown")
    img_data = data["image"].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # YOLO object detection
    results = model(frame)
    detections = results.pandas().xyxy[0]
    labels = detections["name"].tolist()

    cheating = "No"
    for obj in labels:
        if obj not in ALLOWED_OBJECTS:
            cheating = f"Yes ({obj})"
            break

    # Simulated detection values (replace with MediaPipe if needed)
    blink = "Yes" if int(time.time()) % 2 == 0 else "No"
    mouth = "Open" if int(time.time()) % 3 == 0 else "Closed"
    head_pose = ["Left", "Right", "Up", "Down", "Center"][int(time.time()) % 5]

    # Logging
    timestamp = time.strftime('%H:%M:%S')
    report_path = os.path.join("reports", f"user_{user_id}_report.txt")
    with open(report_path, "a") as f:
        log = f"[{timestamp}] Blink: {blink}, Mouth: {mouth}, Head Pose: {head_pose}, Cheating: {cheating}\n"
        f.write(log)

    return jsonify({
        "cheating": cheating,
        "blink": blink,
        "mouth": mouth,
        "head_pose": head_pose
    })

if __name__ == "__main__":
    app.run(debug=True)
