from flask import Flask, request, jsonify, session, redirect, url_for, render_template,send_file
import os
import pdfplumber
import spacy
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import uuid
import numpy as np
import re
import google.generativeai as genai

# Gemini API key setup
genai.configure(api_key="AIzaSyCfFhfqFomEefjAzeq0wF-ldBFC5xw49GY")
model = genai.GenerativeModel('gemini-1.5-pro-latest')

app = Flask(__name__)
app.secret_key = "super_secret_key"
bcrypt = Bcrypt(app)
CORS(app)

UPLOAD_FOLDER = r"C:\MiniProject\backend\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="12345",
        database="resume_analyzer"
    )

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS applicants (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255),
                        email VARCHAR(255) UNIQUE NOT NULL,
                        resume_path VARCHAR(255) NOT NULL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS userapplicants (
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        username VARCHAR(100) NOT NULL,
                        name VARCHAR(100),
                        password VARCHAR(255) NOT NULL,
                        email VARCHAR(150),
                        resume_path VARCHAR(255),
                        quiz_scores VARCHAR(50));''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS user_login (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        applicant_id INT NOT NULL,
                        FOREIGN KEY (applicant_id) REFERENCES applicants(id) ON DELETE CASCADE)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS admin_login (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS quiz_scores (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        applicant_id INT NOT NULL,
                        score FLOAT NOT NULL,
                        FOREIGN KEY (applicant_id) REFERENCES applicants(id) ON DELETE CASCADE)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS resume_scores (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        applicant_id INT NOT NULL,
                        score FLOAT NOT NULL,
                        FOREIGN KEY (applicant_id) REFERENCES applicants(id) ON DELETE CASCADE)''')

    conn.commit()
    conn.close()

init_db()

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

def compute_matching_score(cv_text, cv_entities, required_education, required_skills, required_experience):
    score = 0
    education = cv_entities.get('EDU', '')
    score += fuzz.token_set_ratio(education, required_education) / 100

    for skill in required_skills:
        max_skill_match_score = max([fuzz.token_set_ratio(skill.strip(), word) for word in cv_text.split()] + [0])
        score += max_skill_match_score / 100

    experience_text = cv_entities.get('DATE', '')
    doc1, doc2 = nlp(experience_text), nlp(f"{required_experience} years")
    if doc1.has_vector and doc2.has_vector:
        score += cosine_similarity(doc1.vector.reshape(1, -1), doc2.vector.reshape(1, -1))[0][0]

    return score
def extract_resume_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

# Generate MCQs
def generate_mcqs(resume_text):
    prompt = f"""Analyze the following resume and generate 10 multiple choice questions (MCQs)to test the proficiency of applicant on the technical skills they claim to have on their resume. 
For example if the applicant has claimed to have java programming skills on their resume, ask general questions on jave. Each question should have 4 options and indicate the correct one clearly.

Resume Text:
{resume_text}

Format:
1. Question
A. Option 1
B. Option 2
C. Option 3
D. Option 4
Answer: [Correct Option Letter]

Repeat for 10 questions.
"""
    response = model.generate_content(prompt)
    full_text = response.text

    questions_only = re.sub(r'Answer:\s*[A-D]', '', full_text).strip()
    correct_answers = re.findall(r'Answer:\s*([A-D])', full_text)

    return questions_only, correct_answers

@app.route('/analyze_resumes', methods=['POST'])
def analyze_resumes():
    try:
        if 'resumes' not in request.files:
            return jsonify({"error": "No resumes uploaded"}), 400

        resumes = request.files.getlist('resumes')
        if not resumes:
            return jsonify({"error": "No resumes selected"}), 400

        required_education = request.form.get('required_education', '')
        required_skills_raw = request.form.get('required_skills', '')
        required_experience = request.form.get('required_experience', '0')
        top_cvs_count = int(request.form.get('top_cvs_count', 3))

        required_skills = [s.strip() for s in required_skills_raw.split(',') if s.strip()]
        if not required_skills:
            return jsonify({"error": "Required skills must be provided."}), 400

        cv_scores = {}
        for file in resumes:
            if file.filename.endswith('.pdf'):
                unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(file_path)

                text = extract_text_from_pdf(file_path)
                entities = extract_entities(text)
                score = compute_matching_score(text, entities, required_education, required_skills, required_experience)
                cv_scores[file.filename] = score

        top_cvs = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)[:top_cvs_count]
        return jsonify({"top_resumes": top_cvs})

    except Exception as e:
        return jsonify({"error": "Something went wrong", "details": str(e)}), 500

@app.route('/applicant_login', methods=['POST'])
def applicant_login():
    data = request.json
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    print("Received login attempt:", username, password)

    cursor.execute("SELECT id FROM userapplicants WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session["applicant_id"] = user["id"]
        return jsonify({  # ✅ Return ID for frontend redirect
            "status": "success",
            "id": user["id"]
        })
    else:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401


@app.route('/quiz/<int:applicant_id>', methods=['GET'])
def quiz_pages(applicant_id):
    if "applicant_id" not in session or session["applicant_id"] != applicant_id:
        return "Unauthorized", 403
    conn = mysql.connector.connect(
        host="localhost",         # or your DB host
        user="root",     # your DB username
        password="12345", # your DB password
        database="resume_analyzer"   # your database name
    )
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM userapplicants WHERE id = %s", (applicant_id,))
    result = cursor.fetchone()
    conn.close()

    applicant_name = result[0] if result else "Unknown"

    return render_template("quiz.html", applicant_id=applicant_id, applicant_name=applicant_name)


@app.route('/profile/<int:applicant_id>', methods=['GET'])
def applicant_profile(applicant_id):
    if "applicant_id" not in session or session["applicant_id"] != applicant_id:
        return jsonify({"message": "Unauthorized access"}), 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT u.username, a.resume_path
        FROM user_login u
        LEFT JOIN applicants a ON u.id = a.applicant_id
        WHERE u.id = %s
    """, (applicant_id,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"message": "User not found"}), 404

    return jsonify({
        "username": user["username"] if user["username"] else "Unknown",
        "resume": user["resume_path"] if user["resume_path"] else None
    })

@app.route('/hr_login', methods=['POST'])
def hr_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM admin_login WHERE username = %s", (username,))
    hr_admin = cursor.fetchone()
    conn.close()

    if not hr_admin:
        return jsonify({"message": "HR Admin not found", "status": "error"}), 401

    stored_password = hr_admin["password"]
    if stored_password == password:
        session["hr_id"] = hr_admin["id"]
        return jsonify({"message": "HR Login successful", "status": "success"}), 200
    else:
        return jsonify({"message": "Incorrect password", "status": "error"}), 401

@app.route('/candidate_signup', methods=['POST'])
def candidate_signup():
    data = request.get_json()
    name = data.get("fullname")
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    if not name or not email or not username or not password or not confirm_password:
        return jsonify({"message": "All fields are required", "status": "error"}), 400

    if password != confirm_password:
        return jsonify({"message": "Passwords do not match", "status": "error"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if username already exists
        cursor.execute("SELECT id FROM userapplicants WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({"message": "Username already exists", "status": "error"}), 400

        # Insert the new user data into the table
        cursor.execute("""
            INSERT INTO userapplicants (name, email, username, password)
            VALUES (%s, %s, %s, %s)
        """, (name, email, username, password))

        conn.commit()
        return jsonify({"message": "Signup successful!", "status": "success"}), 200

    except mysql.connector.Error as e:
        return jsonify({"message": "Database error", "status": "error", "error": str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({"message": "No file uploaded", "status": "error"}), 400

    file = request.files['resume']
    name = request.form.get('name')
    email = request.form.get('email')

    if not name or not email:
        return jsonify({"message": "Name and Email are required", "status": "error"}), 400

    if file.filename == "" or not file.filename.endswith('.pdf'):
        return jsonify({"message": "Invalid file type. Only PDF allowed.", "status": "error"}), 400

    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(file_path)

    # ✅ Copy the file to frontend/src directory too
    secondary_path = os.path.join(r"C:\MiniProject\frontend\src", unique_filename)
    with open(file_path, 'rb') as src_file:
        with open(secondary_path, 'wb') as dst_file:
            dst_file.write(src_file.read())

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO applicants (name, email, resume_path) VALUES (%s, %s, %s)", (name, email, unique_filename))
        conn.commit()
        return jsonify({"message": "Resume uploaded successfully!", "status": "success"}), 200
    except mysql.connector.Error as err:
        conn.rollback()
        return jsonify({"message": f"Database error: {err}", "status": "error"}), 500
    finally:
        cursor.close()
        conn.close()


@app.route('/')
def quiz_page():
    return render_template('splash.html')

@app.route('/landingpage')
def landing():
    return render_template('landingpage.html')

@app.route('/loginchoice')
def login_choice():
    return render_template('loginchoice.html')

@app.route('/usersignup')
def user_signup():
    return render_template('usersignup.html')

@app.route('/hrlogin')
def hrlogin():
    return render_template('hrlogin.html')

# Route for Candidate login
@app.route('/userlogin')
def user_login():
    return render_template('userlogin.html')

@app.route('/adminwork')
def admin_login():
    return render_template('adminwork.html')

@app.route('/hrdashboard')
def hrdashboard():
    return render_template('hrdashboard.html')

@app.route('/quiz')
def quizgo():
    return render_template('quiz.html')

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    user_id = request.form.get('user_id')
    name = request.form.get('name')
    file = request.files.get('resume')

    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid or missing PDF file."}), 400


    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    conn = mysql.connector.connect(
        host="localhost",         # or your DB host
        user="root",     # your DB username
        password="12345", # your DB password
        database="resume_analyzer"   )
    
    cursor = conn.cursor()
    cursor.execute("UPDATE userapplicants SET resume_path = %s WHERE id = %s", (filename, user_id))
    conn.commit()

    resume_text = extract_text_from_pdf(file_path)
    prompt = f"""Analyze the following resume and generate 10 multiple choice questions (MCQs)to test the proficiency of applicant on the technical skills they claim to have on their resume. 
                    For example if the applicant has claimed to have java programming skills on their resume, ask general questions on jave. Each question should have 4 options and indicate the correct one clearly.

                Resume Text:
                {resume_text}

                Format:
                1. Question
                A. Option 1
                B. Option 2
                C. Option 3
                D. Option 4
                Answer: [Correct Option Letter]

                Repeat for 10 questions."""
    response = model.generate_content(prompt)
    full_text = response.text

    questions_only = re.sub(r'Answer:\s*[A-D]', '', full_text).strip()
    correct_answers = re.findall(r'Answer:\s*([A-D])', full_text)

    with open(f'{user_id}_questions.txt', 'w', encoding='utf-8') as f:
        f.write(questions_only)

    with open(f'{user_id}_answers.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(correct_answers))

    return jsonify({"questions": questions_only})

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    user_id = request.form.get('user_id')
    name = request.form.get('name')
    user_answers = request.form.get('answers').strip().split()

    try:
        with open(f'{user_id}_answers.txt', 'r', encoding='utf-8') as f:
            correct_answers = f.read().strip().split()
    except FileNotFoundError:
        return jsonify({"error": "Answer key not found for this user."}), 404

    score = sum(1 for u, c in zip(user_answers, correct_answers) if u.upper() == c.upper())

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE userapplicants SET quiz_scores = %s WHERE id = %s", (score,user_id))
    conn.commit()
    conn.close()

    return jsonify({"score": score})
@app.route("/applicants", methods=["GET"])
def get_applicants():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = """SELECT id,name,email,resume_path, quiz_scores
                    FROM userapplicants ;"""
        cursor.execute(query)
        applicants = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(applicants)

    except Exception as e:
        print("Error fetching applicants:", e)
        return jsonify({"error": "Failed to fetch applicant data"}), 500

@app.route("/open_resume/<int:user_id>", methods=["GET"])
def open_resume(user_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT resume_path FROM userapplicants WHERE id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            filename = os.path.basename(result[0])  # extract just the filename
            full_path = os.path.join(UPLOAD_FOLDER, filename)

            if os.path.exists(full_path):
                return send_file(full_path)
            else:
                return jsonify({"error": "File not found"}), 404
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        print("Error opening resume:", e)
        return jsonify({"error": "Failed to open resume"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
