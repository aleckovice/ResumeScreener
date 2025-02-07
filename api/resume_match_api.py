import sys
import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from utils.resume_parser import extract_and_clean_text
from models.match_resume import match_resume_to_job
from models.classify_resume import classify_resume

# Ensure Python finds the correct project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Explicitly set the absolute path to templates folder
TEMPLATES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))

app = Flask(__name__, template_folder=TEMPLATES_DIR)

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/match", methods=["POST"])
def match_resume():
    if "resume" not in request.files or "job_description" not in request.form:
        return jsonify({"error": "Missing resume file or job description"}), 400

    resume_file = request.files["resume"]
    job_description = request.form["job_description"]

    filename = secure_filename(resume_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    resume_file.save(filepath)

    # Compute match score
    result = match_resume_to_job(filepath, job_description)

    return jsonify(result)

@app.route("/classify", methods=["POST"])
def classify_resume_api():
    if "resume" not in request.files:
        return jsonify({"error": "Missing resume file"}), 400

    resume_file = request.files["resume"]
    filename = secure_filename(resume_file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    resume_file.save(filepath)

    # Classify resume
    category = classify_resume(filepath)

    return jsonify({"category": category})

if __name__ == "__main__":
    app.run(debug=True)
