import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils.resume_parser import process_all_resumes, extract_and_clean_text

# Load pre-trained SBERT model
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    """Generates SBERT embeddings for a list of texts."""
    return sbert_model.encode(texts, convert_to_tensor=False)

def tune_hyperparameters(model, param_grid, X_train, y_train):
    """Performs hyperparameter tuning using GridSearchCV."""
    print(f"Tuning hyperparameters for {model.__class__.__name__}...")
    grid_search = GridSearchCV(model, param_grid, cv=4, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_resume_classifier(data_dir="data/"):
    """Trains a classification model to categorize resumes by job field with hyperparameter tuning."""
    print("Extracting and processing resumes...")
    resume_texts, labels = process_all_resumes(data_dir)
    
    print("Generating embeddings...")
    X = get_embeddings(resume_texts)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing classifiers and hyperparameter grids...")
    classifiers = {
        "Logistic Regression": (LogisticRegression(), {'C': [0.1, 1, 10], 'max_iter': [200, 500, 1000]}),
        "Random Forest": (RandomForestClassifier(random_state=42), {'n_estimators': [5, 10, 25, 50, 100, 200, 500], 'max_depth': [10, 20, None]}),
        "SVM": (SVC(kernel='linear'), {'C': [0.1, 1, 10]})
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, (model, param_grid) in classifiers.items():
        tuned_model = tune_hyperparameters(model, param_grid, X_train, y_train)
        accuracy = tuned_model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.2f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = tuned_model
    
    print(f"Best Model: {best_model.__class__.__name__} with accuracy {best_accuracy:.2f}")
    
    # Save best model
    with open("models/resume_classifier.pkl", "wb") as model_file:
        pickle.dump(best_model, model_file)
    
    print("Model training complete. Saved as models/resume_classifier.pkl")

def classify_resume(resume_path, model_path="models/resume_classifier.pkl"):
    """Classifies a resume into a job category."""
    with open(model_path, "rb") as model_file:
        classifier = pickle.load(model_file)
    
    resume_text = extract_and_clean_text(resume_path)  # Extract text
    resume_embedding = get_embeddings([resume_text])  # Convert to embedding
    predicted_category = classifier.predict(resume_embedding)[0]  # Predict category
    
    return predicted_category

# Example Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:  # If a file path is provided, classify the resume
        resume_path = sys.argv[1]
        category = classify_resume(resume_path)
        print("Predicted Category:", category)
    else:
        train_resume_classifier()
