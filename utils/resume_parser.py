import fitz  # PyMuPDF for PDFs
import docx
import os
import nltk
from nltk.tokenize import sent_tokenize
from utils.text_preprocessing import clean_text

nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_relevant_resume_sections(resume_text):
    """Extracts key sections from a resume like Skills, Experience, and Certifications."""
    sections_to_keep = []
    keywords = ['skills', 'experience', 'certifications', 'education', 'summary']
    sentences = sent_tokenize(resume_text.lower())
    
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                sections_to_keep.append(sentence)
                break  # Avoid duplicating sentences if multiple keywords match
    
    return ' '.join(sections_to_keep)

def extract_and_clean_text(file_path):
    """Extracts and cleans text from a resume file, keeping key sections."""
    if file_path.endswith(".pdf"):
        raw_text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        raw_text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are allowed.")
    
    filtered_text = extract_relevant_resume_sections(raw_text)  # Keep key sections
    return clean_text(filtered_text)

def process_all_resumes(directory="data/"):
    """Extracts and cleans text from all resumes in categorized job folders."""
    resume_texts = []
    labels = []
    
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):  # Ensure it's a folder
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                try:
                    text = extract_and_clean_text(file_path)
                    resume_texts.append(text)
                    labels.append(category)  # Assign category as label
                except Exception as e:
                    print(f"Error processing {filename} in {category}: {e}")
    
    return resume_texts, labels

# Example Usage
if __name__ == "__main__":
    print("Processing all resumes from categorized job folders...")
    resume_texts, labels = process_all_resumes("data/")
    print(f"Extracted {len(resume_texts)} resumes.")
    print("Sample:", resume_texts[:1], "Label:", labels[:1])
