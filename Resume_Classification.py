# Import Required libraries
import streamlit as st
import pickle
from PyPDF2 import PdfReader
import docx
import subprocess
import tempfile
import re
import nltk
import os
import glob
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Load trained ML files
model = pickle.load(open("svm_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# Text cleaning
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n',' ',text)
    text = re.sub(r'\d+',' NUM ',text)
    text = re.sub(r'[^a-zA-Z ]',' ',text)
    text = re.sub(r'\s+',' ',text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Skills List
skills_list = [
    "java","c++","c","shell scripting",
    "html","css","javascript","typescript","jquery","react",
    "angular","bootstrap","redux","saga","thunk",
    "node js","express js","mongodb","material ui",
    "sql","mysql","postgresql","oracle db",
    "git","github","jenkins","ansible","docker","kubernetes",
    "linux","unix","windows","vs code",
    "winscp","filezilla","pcomm","tws","service now","ms office",
    "xml","xslt","json","soap","rest api","web services",
    "peoplesoft","peoplesoft hrms","fscm"
]
# Skills extraction Function
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return list(set(found_skills))

# EXperience extraction function
def extract_experience(text):
    text = text.lower()
    patterns = [
        r'(\d+)\+?\s+years',
        r'(\d+)\s+yrs',
        r'(\d+)\s+year'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1) + " years"
    return "Not Found"

# pdf text extraction function
def extract_text_from_pdf(path):
    text = ""
    pdf = PdfReader(path)
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Docx text extraction function
def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# DOC → DOCX conversion function
def convert_doc_to_docx(file_path):
    libreoffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
    output_dir = os.path.dirname(file_path)

    for f in glob.glob(os.path.join(output_dir, "*.docx")):
        try:
            os.remove(f)
        except:
            pass

    subprocess.run([
        libreoffice_path,
        "--headless",
        "--convert-to",
        "docx",
        file_path,
        "--outdir",
        output_dir
    ])

    time.sleep(3)

    converted_files = glob.glob(os.path.join(output_dir, "*.docx"))
    if converted_files:
        return converted_files[0]
    else:
        return None

# Streamlit UI
st.title("🤖 Resume Screening System")
st.write("Upload resumes (PDF, DOCX, DOC)")

# File Upload
uploaded_files = st.file_uploader(
    "Upload Resume Files",
    type=["pdf", "docx", "doc"],
    accept_multiple_files=True
)

# MAIN PIPELINE
if st.button("Analyze Resume"):
    if uploaded_files:
        for file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.getbuffer())
                temp_path = temp_file.name

            if file.name.endswith(".doc"):
                converted = convert_doc_to_docx(temp_path)
                if converted is None:
                    st.error(f"Failed to convert {file.name}")
                    continue
                temp_path = converted

            if file.name.endswith(".pdf"):
                resume_text = extract_text_from_pdf(temp_path)
            else:
                resume_text = extract_text_from_docx(temp_path)

            cleaned_text = clean_text(resume_text)

            # Job Role Prediction
            resume_tfidf = tfidf.transform([cleaned_text])
            pred_label = model.predict(resume_tfidf)
            job_role = le.inverse_transform(pred_label)[0]

            # Skills + Experience
            skills = extract_skills(resume_text)
            experience = extract_experience(resume_text)

            # Display Output
            st.success(f"📄 {file.name}")
            st.write(f"**Predicted Role:** {job_role}")
            st.write(f"**Skills Found:** {', '.join(skills) if skills else 'None'}")
            st.write(f"**Experience:** {experience}")
            st.divider()

    else:
        st.warning("Please upload at least one resume.")