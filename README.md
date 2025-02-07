# Resume Screener

## Overview
Resume Screener is a web-based application that allows users to:
- **Match resumes to job descriptions** using **SBERT-based semantic similarity**.
- **Classify resumes into job categories** using a **machine learning classifier**.
- **Upload resumes (PDF/DOCX)** and receive real-time analysis.

## Features
- **Resume Matching**: Computes a similarity score between a resume and a job description.
- **Resume Classification**: Predicts the job category based on resume content.
- **Flask API**: Handles requests for resume processing.
- **Machine Learning Model**:
  - **SBERT (Sentence-BERT) embeddings** for feature extraction.
  - **Logistic Regression, Random Forest, and SVM** for classification.
  - **Hyperparameter tuning with GridSearchCV**.
- **Bootstrap UI**: Clean and responsive interface.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/aleckovice/ResumeScreener.git
cd ResumeScreener
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
conda create --name resume_screener python=3.9 -y
conda activate resume_screener
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask API
```bash
python -m api.resume_match_api
```

### 5. Access the Web Interface
- Open a browser and navigate to: `http://127.0.0.1:5000`
- Upload a resume and:
  - **Check Job Fit** (Match resume with job description).
  - **Classify Resume** (Predict the job category).

## Directory Structure
```
resume-screener/
│── api/
│   ├── resume_match_api.py  # Flask API handling requests
│── models/
│   ├── train_model.py       # Handles resume-job matching
│   ├── classify_resume.py   # Handles resume classification
│── utils/
│   ├── resume_parser.py     # Extracts and cleans resume text
│   ├── text_preprocessing.py # Preprocessing for both resumes and job descriptions
│── templates/
│   ├── index.html           # Frontend UI
│── data/
│   ├── ENGINEERING/         # Sample resumes categorized by job fields
│── uploads/                 # Stores uploaded resumes
│── README.md                # Documentation
│── requirements.txt         # Dependencies
```

## Future Improvements
- **Improve classification accuracy** using deep learning models (e.g., BERT fine-tuning).
- **Enhance job matching** with domain-specific similarity techniques.


## License
This project is open-source and available under the [MIT License](LICENSE).

## Contributors
- **Your Name** ([@aleckovice](https://github.com/aleckovice))

## Acknowledgments
- Built using **Flask, SBERT, Scikit-Learn, and Bootstrap**.

