from sentence_transformers import SentenceTransformer, util
from utils.resume_parser import extract_and_clean_text
from utils.text_preprocessing import extract_key_sections

# Load pre-trained SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_sbert_similarity(resume_text, job_description):
    """Computes semantic similarity using SBERT embeddings."""
    resume_embedding = sbert_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = sbert_model.encode(job_description, convert_to_tensor=True)
    
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return similarity_score

def match_resume_to_job(resume_path, job_desc, threshold=0.5):
    """Matches a single resume to a job description using SBERT similarity."""
    resume_text = extract_and_clean_text(resume_path)  # Extract and clean resume text
    job_desc_cleaned = extract_key_sections(job_desc)  # Extract key sections from job description
    
    similarity_score = compute_sbert_similarity(resume_text, job_desc_cleaned)
    is_good_fit = similarity_score >= threshold
    
    return {
        "similarity_score": round(similarity_score, 4),
        "good_fit": bool(is_good_fit)  # Convert numpy.bool_ to Python bool
    }

# Example Usage
if __name__ == "__main__":
    sample_resume = "data/ENGINEERING/12518008.pdf"  # Replace with an actual resume file
    sample_job_desc = "Looking for a software engineer proficient in Python and machine learning."
    
    result = match_resume_to_job(sample_resume, sample_job_desc)
    print("Similarity Score:", result["similarity_score"])
    print("Good Fit:", result["good_fit"])
