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
import pandas as pd
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

# Skills extraction function
def extract_skills(text):
    text = text.lower()
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return list(set(found_skills))

# Experience extraction function
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

# PDF extraction
def extract_text_from_pdf(path):
    text = ""
    pdf = PdfReader(path)
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# DOCX extraction
def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

# DOC → DOCX conversion
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
st.title("🤖 Resume Screening Dashboard")
st.write("Upload resumes (PDF, DOCX, DOC)")

uploaded_files = st.file_uploader(
    "Upload Resume Files",
    type=["pdf", "docx", "doc"],
    accept_multiple_files=True
)

# MAIN PIPELINE
if st.button("Analyze Resume"):
    if uploaded_files:

        results = []

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

            results.append({
                "File Name": file.name,
                "Predicted Role": job_role,
                "Skills": ", ".join(skills) if skills else "None",
                "Experience": experience
            })

        # Create DataFrame
        df_results = pd.DataFrame(results)

        # Convert experience text → number
        def exp_to_num(exp):
            try:
                return int(exp.split()[0])
            except:
                return 0

        df_results["Exp_num"] = df_results["Experience"].apply(exp_to_num)

        # Sort results
        df_results = df_results.sort_values(
            by=["Predicted Role", "Exp_num"],
            ascending=[True, False]
        )

        df_results.drop("Exp_num", axis=1, inplace=True)
        df_results.reset_index(drop=True, inplace=True)

        # Show table in app
        st.subheader("📊 Resume Screening Results")
        st.dataframe(df_results, use_container_width=True)

        # Excel download
        excel_file = "Resume_Screening_Results.xlsx"
        df_results.to_excel(excel_file, index=False)

        with open(excel_file, "rb") as f:
            st.download_button(
                label="Download Results as Excel",
                data=f,
                file_name=excel_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.warning("Please upload at least one resume.")
