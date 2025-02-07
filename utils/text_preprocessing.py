import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """Cleans and preprocesses text by removing special characters, stopwords, and unnecessary spaces."""
    text = text.lower()
    text = re.sub(r'\n', ' ', text)  # Remove new lines
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(words)

def extract_key_sections(job_description):
    """Extracts key sections from job descriptions to improve relevance."""
    sections_to_keep = []
    keywords = ['job summary', 'responsibilities', 'qualifications', 'requirements', 'skills', 'experience', 'license', 'certification']
    sentences = sent_tokenize(job_description.lower())
    
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                sections_to_keep.append(sentence)
                break  # Avoid duplicating sentences if multiple keywords match
    
    return ' '.join(sections_to_keep)

# Example Usage
if __name__ == "__main__":
    sample_job_desc = """
    Job Summary: Implements and creates engaging lessons that align with the curriculum.
    Qualifications: Must have a PEL with ESL endorsement.
    Experience: At least 1 year of teaching experience preferred.
    Benefits: Health insurance, dental insurance, professional development.
    """
    cleaned_text = extract_key_sections(sample_job_desc)
    print("Filtered Job Description:", cleaned_text)