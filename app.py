# app.py
from flask import Flask, render_template, request, redirect, session, jsonify, send_file, url_for
import sqlite3
import base64
import numpy as np
import cv2
import time
import os
import bcrypt
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from datetime import datetime
from emotion_detection import detect_emotion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "exam_system.db")

try:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except:
    model = None

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

os.makedirs("reports", exist_ok=True)
os.makedirs("certificates", exist_ok=True)


# ================= PERFORMANCE INSIGHTS =================
def calculate_performance_insights(answers, questions, cheating_count=0):
    subject_stats = {}

    for q in questions:
        subject = q["subject_name"]   # ✅ REAL NAME

        if subject not in subject_stats:
            subject_stats[subject] = {"total": 0, "correct": 0}

        subject_stats[subject]["total"] += 1

        selected = answers.get(f"q{q['id']}")
        if selected and int(selected) == int(q["correct_answer"]):
            subject_stats[subject]["correct"] += 1

    insights = []
    tips = []

    for subject, data in subject_stats.items():
        percentage = int((data["correct"] / data["total"]) * 100)

        if percentage >= 80:
            level = "Good"
        elif percentage >= 50:
            level = "Average"
            tips.append(f"Revise core concepts in {subject}.")
        else:
            level = "Weak"
            tips.append(f"Practice more questions in {subject}.")

        insights.append({
            "subject": subject,
            "percentage": percentage,
            "level": level
        })

    if cheating_count > 0:
        tips.append("Avoid distractions and follow exam rules strictly.")

    return insights, tips

# ---------------- DB ----------------
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = bcrypt.hashpw(request.form["password"].encode(), bcrypt.gensalt())
        role = request.form.get("role", "student")

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?,?,?)",
                (username, password, role)
            )
            conn.commit()
        except:
            return "Username exists"
        finally:
            conn.close()
        return redirect("/login")

    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"].encode()

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()
        conn.close()

        if user and bcrypt.checkpw(password, user["password"]):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect("/admin")

            # ================= MULTI SUBJECT FIX =================
            conn = get_db_connection()
            subjects = conn.execute(
                "SELECT id FROM subjects ORDER BY id"
            ).fetchall()
            conn.close()

            session["subject_queue"] = [s["id"] for s in subjects]
            session["completed_subjects"] = []
            # =====================================================

            return redirect(f"/exam/{session['subject_queue'][0]}")

        return "Invalid login"

    return render_template("login.html")

# ---------------- ADMIN ----------------
@app.route("/admin", methods=["GET","POST"])
def admin():
    if session.get("role") != "admin":
        return redirect("/")

    conn = get_db_connection()
    cur = conn.cursor()

    subjects = cur.execute("SELECT * FROM subjects").fetchall()

    if request.method == "POST":
        cur.execute("""
            INSERT INTO questions
            (subject_id, question, option1, option2, option3, option4, correct_answer)
            VALUES (?,?,?,?,?,?,?)
        """, (
            request.form["subject_id"],
            request.form["question"],
            request.form["option1"],
            request.form["option2"],
            request.form["option3"],
            request.form["option4"],
            int(request.form["correct_answer"])
        ))
        conn.commit()

    questions = cur.execute("SELECT * FROM questions").fetchall()
    conn.close()

    return render_template("admin_dashboard.html",
                           questions=questions,
                           subjects=subjects)

# ---------------- EXAM ----------------
@app.route("/exam/<int:subject_id>")
def exam(subject_id):
    if "user_id" not in session:
        return redirect("/login")

    if subject_id not in session.get("subject_queue", []):
       # ALL SUBJECTS COMPLETED
       return redirect("/result")

    conn = get_db_connection()
    questions = conn.execute(
        "SELECT * FROM questions WHERE subject_id=?",
        (subject_id,)
    ).fetchall()
    subject = conn.execute(
        "SELECT * FROM subjects WHERE id=?",
        (subject_id,)
    ).fetchone()
    conn.close()

    return render_template(
        "exam.html",
        questions=questions,
        subject=subject,
        exam_time=15 * 60
    )

# ---------------- SUBMIT EXAM ----------------
# ---------------- SUBMIT EXAM ----------------
@app.route("/submit_exam/<int:subject_id>", methods=["POST"])
def submit_exam(subject_id):
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()
    cur = conn.cursor()

    questions = cur.execute("""
    SELECT q.*, s.name AS subject_name
    FROM questions q
    JOIN subjects s ON q.subject_id = s.id
    WHERE q.subject_id=?
""", (subject_id,)).fetchall()


    # 2️⃣ Calculate SCORE
    score = 0
    for q in questions:
        selected = request.form.get(f"q{q['id']}")
        if selected and int(selected) == int(q["correct_answer"]):
            score += 1

    # 3️⃣ TOTAL & PERCENTAGE
    total_questions = len(questions)
    percentage = round((score / total_questions) * 100, 2) if total_questions else 0

    # 4️⃣ SAVE RESULT
    cur.execute("""
        INSERT INTO exam_results (user_id, subject_id, score, total, time_taken)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        subject_id,
        score,
        total_questions,
        "15 mins"
    ))

    conn.commit()
    conn.close()

    # ✅ 5️⃣ PERFORMANCE INSIGHTS
    insights, tips = calculate_performance_insights(
        request.form,
        questions,
        cheating_count=0
    )

    # 6️⃣ Initialize once
    if "insights" not in session:
        session["insights"] = []

    if "tips" not in session:
        session["tips"] = []

    # 7️⃣ Append (multi-subject support)
    session["insights"].extend(insights)

    for tip in tips:
        if tip not in session["tips"]:
            session["tips"].append(tip)

    # 8️⃣ MULTI SUBJECT FLOW
    session["completed_subjects"].append(subject_id)
    session["subject_queue"].remove(subject_id)
    session.modified = True

    if session["subject_queue"]:
        return redirect(f"/exam/{session['subject_queue'][0]}")
    else:
        return redirect("/result")

    # =====================================================
# TEMP DEBUG: Confirm existing tables
@app.route("/debug_tables")
def debug_tables():
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    # Return table names as a simple list
    return "<br>".join([t[0] for t in tables])

# ---------------- FINAL RESULT ----------------
@app.route("/result")
def result():

    insights = session.get("insights", [])
    tips = session.get("tips", [])
    if "user_id" not in session:
        return redirect("/login")

    conn = get_db_connection()

    # ---------------- SUBJECT-WISE (for charts) ----------------
    subject_rows = conn.execute("""
        SELECT s.name AS subject_name, e.score, e.total
        FROM exam_results e
        JOIN subjects s ON s.id = e.subject_id
        WHERE e.user_id = ?
        GROUP BY s.id
        ORDER BY s.id
    """, (session["user_id"],)).fetchall()

    total_score = sum(r["score"] for r in subject_rows)
    total_questions = sum(r["total"] for r in subject_rows)

    percentage = round((total_score / total_questions) * 100, 2) if total_questions else 0

    # ---------------- CERTIFICATE TYPE ----------------
    if percentage >= 90:
        certificate_type = "excellent"
        message = "Outstanding! 🏆 Perfect Score!"
    elif percentage >= 70:
        certificate_type = "excellent"
        message = "Excellent Performance!"
    elif percentage >= 50:
        certificate_type = "good"
        message = "Good Job!"
    else:
        certificate_type = "improvement"
        message = "Needs Improvement"

    # ---------------- SAVE ONLY ONCE ----------------
    existing = conn.execute(
        "SELECT id FROM results WHERE user_id = ?",
        (session["user_id"],)
    ).fetchone()

    if not existing:
        conn.execute("""
            INSERT INTO results (user_id, score, total, percentage, certificate_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session["user_id"],
            total_score,
            total_questions,
            percentage,
            certificate_type,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()

    conn.close()

    return render_template(
        "result.html",
        subjects=subject_rows,
        total_score=total_score,
        total_questions=total_questions,
        percentage=percentage,
        message=message,
        insights=insights,
        tips=tips   
    )

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- Cheating Detection ----------------
@app.route("/detect_cheating", methods=["POST"])
def detect_cheating():
    data = request.get_json()
    user_id = data.get("user_id", "unknown")

    # DEFAULT VALUES
    cheating = "No"
    blink = "No"
    mouth = "Closed"
    head_pose = "Center"
    emotion = "Neutral"
    emotion_conf = 0
    object_status = "Normal"
    audio_status = "Normal"

    try:
        if model is None:
            raise Exception("YOLO model not loaded")

        img_data = data["image"].split(",")[1]
        frame = cv2.imdecode(
            np.frombuffer(base64.b64decode(img_data), np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise Exception("Empty frame")

        # ✅ YOLO
        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            if row["name"] not in ["person"] and row["confidence"] > 0.20:
                cheating = "Yes"
                object_status = f"{row['name']} detected"
                break

        # TEMP LOGIC
        blink = "Yes" if int(time.time()) % 2 == 0 else "No"
        mouth = "Open" if int(time.time()) % 3 == 0 else "Closed"
        head_pose = ["Left","Right","Up","Down","Center"][int(time.time()) % 5]

        e = detect_emotion(frame)
        if e:
            emotion, emotion_conf = e

    except Exception as err:
        print("DETECTION ERROR:", err)

    # LOG
    timestamp = time.strftime('%H:%M:%S')
    with open(f"reports/user_{user_id}_report.txt", "a") as f:
        f.write(f"[{timestamp}] Cheating: {cheating}, Object: {object_status}\n")

    return jsonify({
        "cheating": cheating,
        "blink": blink,
        "mouth": mouth,
        "head_pose": head_pose,
        "emotion": emotion,
        "emotion_conf": emotion_conf,
        "object": object_status,
        "audio": audio_status
    })

# ---------------- Cheating Log ----------------
@app.route("/log_cheating", methods=["POST"])
def log_cheating():
    if "user_id" not in session:
        return jsonify({"status": "error"}), 401

    data = request.get_json()
    incident_type = data.get("type", "Unknown")
    user_id = data.get("user_id", session["user_id"])

    timestamp = time.strftime('%H:%M:%S')
    report_path = os.path.join("reports", f"user_{user_id}_report.txt")

    with open(report_path, "a") as f:
        f.write(f"[{timestamp}] Cheating Detected: {incident_type}\n")

    return jsonify({"status": "success"})
# ---------------- Certificate Helpers ----------------
def clamp_percentage(raw):
    try:
        pct = float(raw)
    except Exception:
        pct = 0.0
    return max(0, min(100, round(pct, 2)))

def draw_certificate_pdf(buffer, username, pct, template='excellent'):
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    width, height = landscape(A4)
    margin = 2*cm

    if template == 'excellent':
        border = (0.06, 0.45, 0.14)
        subtitle = "Outstanding Achievement"
    elif template == 'good':
        border = (0.85, 0.45, 0.08)
        subtitle = "Certificate of Merit"
    else:
        border = (0.6, 0.08, 0.15)
        subtitle = "Certificate of Participation"

    c.setStrokeColorRGB(*border)
    c.setLineWidth(4)
    c.rect(margin/2, margin/2, width - margin, height - margin)

    c.setFillColorRGB(*border)
    c.setFont("Helvetica-Bold", 34)
    c.drawCentredString(width/2, height - 3*cm, subtitle)

    c.setFillColorRGB(0,0,0)
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height - 4.5*cm, "This certifies that")

    c.setFillColorRGB(*border)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width/2, height - 6.5*cm, username)

    c.setFillColorRGB(0,0,0)
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height - 8.2*cm,
                        f"has completed the online exam with a score of {pct:.2f}%.")

    if pct >= 90:
        grade = "Distinction"
    elif pct >= 70:
        grade = "Excellent"
    elif pct >= 50:
        grade = "Good"
    else:
        grade = "Needs Improvement"

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 9.5*cm, f"Grade: {grade}")

    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height - 11*cm, f"Date: {datetime.now().strftime('%d %B %Y')}")

    sig_y = 3.5*cm
    c.drawString(100, sig_y + 20, "Examiner")
    c.line(100, sig_y + 15, 260, sig_y + 15)

    c.drawString(width - 260, sig_y + 20, "Authorized Signatory")
    c.line(width - 260, sig_y + 15, width - 80, sig_y + 15)

    c.showPage()
    c.save()
    buffer.seek(0)

# ---------------- Certificate Download ----------------
@app.route("/download_certificate")
def download_certificate():
    if "user_id" not in session:
        return redirect("/login")

    username = session.get("username", "Student")
    raw_score = request.args.get("score")

    if raw_score is None:
        try:
            conn = get_db_connection()
            row = conn.execute(
                "SELECT percentage FROM results WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (session["user_id"],)
            ).fetchone()
            conn.close()
            raw_score = row["percentage"] if row else 0
        except:
            raw_score = 0

    pct = clamp_percentage(raw_score)

    if pct >= 70:
        template = "excellent"
    elif pct >= 50:
        template = "good"
    else:
        template = "improvement"

    buffer = BytesIO()
    draw_certificate_pdf(buffer, username, pct, template)

    filename = f"{username}_certificate_{int(pct)}.pdf"
    try:
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")
    except TypeError:
        return send_file(buffer, as_attachment=True, attachment_filename=filename, mimetype="application/pdf")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
